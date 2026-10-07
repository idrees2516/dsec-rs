//! Parquet dataset loading (feature `parquet`).
//!
//! Reads the actual HuggingFace parquet shards
//! (`code.parquet`, `cyber.parquet`, `general/train.parquet`,
//! `music.parquet`, `webdev.parquet`) of
//! `XiaomiMiMo/MiMo-V2.6-RL-oss` into [`TaskDataset`] rows.
//!
//! The path is arrow-based: parquet → `RecordBatch` → line-delimited
//! JSON (arrow's JSON writer) → `TaskRow` (serde). Round-trip tests
//! build the parquet fixture with arrow's JSON reader, so the writer
//! and reader sides exercise the same schema shape:
//!
//! ```text
//! prompt        list<struct<role, content>>
//! data_source   utf8
//! ability       utf8
//! agent_name    utf8
//! reward_model  struct<style, ground_truth>
//! extra_info    struct<index, instance_id, dataset_type, instance_json, ...>
//! ```

use crate::error::{Error, Result};
use crate::task::{TaskDataset, TaskRow};

fn arrow_err(e: arrow::error::ArrowError) -> Error {
    Error::InvalidSpec(format!("arrow: {e}"))
}

/// Reads every row of one parquet shard.
pub fn load_parquet(path: impl AsRef<std::path::Path>) -> Result<TaskDataset> {
    // `File` implements parquet's ChunkReader directly
    let file = std::fs::File::open(path.as_ref())?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::InvalidSpec(format!("parquet reader: {e}")))?;
    let batch_reader = builder
        .build()
        .map_err(|e| Error::InvalidSpec(format!("parquet build: {e}")))?;
    let batches: std::result::Result<Vec<arrow::record_batch::RecordBatch>, _> =
        batch_reader.into_iter().collect();
    let batches = batches.map_err(|e| Error::InvalidSpec(format!("parquet decode: {e}")))?;
    load_parquet_arrow(batches)
}

/// Reads every row via the arrow JSON bridge.
pub fn load_parquet_arrow(
    batches: impl IntoIterator<Item = arrow::record_batch::RecordBatch>,
) -> Result<TaskDataset> {
    let mut rows = Vec::new();
    for batch in batches {
        let mut buf = Vec::new();
        {
            use arrow::json::LineDelimitedWriter;
            let mut writer = LineDelimitedWriter::new(&mut buf);
            writer.write(&batch).map_err(arrow_err)?;
            writer.finish().map_err(arrow_err)?;
        }
        for line in String::from_utf8_lossy(&buf).lines() {
            if line.trim().is_empty() {
                continue;
            }
            rows.push(serde_json::from_str::<TaskRow>(line)?);
        }
    }
    Ok(TaskDataset { rows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int64Array, ListArray, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn code_schema() -> Schema {
        let prompt_fields = Fields::from(vec![
            Field::new("role", DataType::Utf8, true),
            Field::new("content", DataType::Utf8, false),
        ]);
        let reward_fields = Fields::from(vec![
            Field::new("style", DataType::Utf8, false),
            Field::new("ground_truth", DataType::Utf8, true),
        ]);
        let extra_fields = Fields::from(vec![
            Field::new("index", DataType::Int64, true),
            Field::new("instance_id", DataType::Utf8, true),
            Field::new("dataset_type", DataType::Utf8, true),
            Field::new("instance_json", DataType::Utf8, true),
        ]);
        Schema::new(vec![
            Field::new(
                "prompt",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Struct(prompt_fields),
                    true,
                ))),
                false,
            ),
            Field::new("data_source", DataType::Utf8, true),
            Field::new("ability", DataType::Utf8, true),
            Field::new("agent_name", DataType::Utf8, true),
            Field::new("reward_model", DataType::Struct(reward_fields), true),
            Field::new("extra_info", DataType::Struct(extra_fields), true),
        ])
    }

    fn code_batch() -> RecordBatch {
        let schema = Arc::new(code_schema());
        // prompt: list<struct{role, content}> — legacy rows omit `role`
        let roles: Vec<Option<&str>> = vec![None, Some("user")];
        let contents: Vec<&str> = vec![
            "[FEAT] Make `secrets` optional.",
            "AddressSanitizer in rfc1035.c",
        ];
        let values = StructArray::from(vec![
            (
                Arc::new(Field::new("role", DataType::Utf8, true)),
                Arc::new(StringArray::from(roles.clone())) as ArrayRef,
            ),
            (
                Arc::new(Field::new("content", DataType::Utf8, false)),
                Arc::new(StringArray::from(contents)) as ArrayRef,
            ),
        ]);
        let offsets = arrow::buffer::OffsetBuffer::from_lengths([1usize, 1usize]);
        let prompt = ListArray::new(
            Arc::new(Field::new("item", values.data_type().clone(), true)),
            offsets,
            Arc::new(values),
            None,
        );
        let data_source = StringArray::from(vec!["opensource-code", "arvo"]);
        let ability = StringArray::from(vec!["swe", "swe"]);
        let agent_name = StringArray::from(vec!["mimo_swe_agent", "mimo_swe_agent"]);
        let reward_model = StructArray::from(vec![
            (
                Arc::new(Field::new("style", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["rule", "rule"])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("ground_truth", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![Some(""), Some("")])) as ArrayRef,
            ),
        ]);
        let extra_model = StructArray::from(vec![
            (
                Arc::new(Field::new("index", DataType::Int64, true)),
                Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("instance_id", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![
                    "format-code-task-001457",
                    "arvo_35858",
                ])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("dataset_type", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec!["opensource-code", "arvo"])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("instance_json", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![
                    r#"{"cwd": "/testbed", "docker_image": "format-code-task-001457:latest", "instance_id": "format-code-task-001457"}"#,
                    r#"{"cwd": "/testbed", "docker_image": "arvo-rl:v1-arvo-35858", "instance_id": "arvo_35858"}"#,
                ])) as ArrayRef,
            ),
        ]);
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(prompt) as ArrayRef,
                Arc::new(data_source) as ArrayRef,
                Arc::new(ability) as ArrayRef,
                Arc::new(agent_name) as ArrayRef,
                Arc::new(reward_model) as ArrayRef,
                Arc::new(extra_model) as ArrayRef,
            ],
        )
        .expect("batch builds")
    }

    #[test]
    fn parquet_roundtrip_through_arrow_json() {
        let batch = code_batch();
        // write to an in-memory parquet, then read back
        let mut buf = Vec::new();
        {
            let schema = batch.schema();
            let mut writer =
                parquet::arrow::arrow_writer::ArrowWriter::try_new(&mut buf, schema, None)
                    .expect("writer");
            writer.write(&batch).expect("write");
            writer.close().expect("close");
        }
        // write to a temp file and read it back through the public API
        let dir =
            std::env::temp_dir().join(format!("dsec-agentenv-parquet-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("code.parquet");
        std::fs::write(&path, &buf).unwrap();
        let ds = load_parquet(&path).unwrap();
        assert_eq!(ds.len(), 2);
        let row = &ds.rows[0];
        assert_eq!(row.data_source, "opensource-code");
        assert_eq!(row.prompt.len(), 1);
        assert_eq!(row.prompt[0].role, "user"); // legacy default (null role)
        assert_eq!(row.prompt[0].content, "[FEAT] Make `secrets` optional.");
        assert_eq!(row.reward_model.style, "rule");
        assert_eq!(row.extra_info.instance_id, "format-code-task-001457");
        let inst = row.instance().unwrap();
        assert_eq!(inst.cwd, "/testbed");
        assert_eq!(inst.docker_image, "format-code-task-001457:latest");
        // the second row keeps its explicit role
        assert_eq!(ds.rows[1].extra_info.instance_id, "arvo_35858");
        assert_eq!(ds.rows[1].prompt[0].role, "user");
    }
}
