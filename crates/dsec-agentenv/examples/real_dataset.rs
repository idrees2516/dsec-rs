//! Loads the real HuggingFace parquet shards of XiaomiMiMo/MiMo-V2.6-RL-oss
//! (downloaded to scripts/research/mimo-samples/) and validates the schema
//! round-trip. Run with:
//! `cargo run -p dsec-agentenv --features parquet --example real_dataset <dir>`

use dsec_agentenv::parquet::load_parquet;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: real_dataset <dir-with-parquet-files>");
    for name in [
        "code.parquet",
        "cyber.parquet",
        "music.parquet",
        "webdev.parquet",
        "general_train.parquet",
    ] {
        let path = std::path::Path::new(&dir).join(name);
        if !path.exists() {
            println!("-- {name}: SKIP (not found)");
            continue;
        }
        match load_parquet(&path) {
            Ok(ds) => {
                println!("-- {}: {} rows", name, ds.len());
                if let Some(row) = ds.rows.first() {
                    println!(
                        "   first: data_source={} ability={} agent={} prompt_len={} instance_id={}",
                        row.data_source,
                        row.ability,
                        row.agent_name,
                        row.prompt.len(),
                        row.extra_info.instance_id
                    );
                    let inst = row
                        .instance()
                        .map(|i| (i.cwd.clone(), i.docker_image.clone()));
                    println!("   instance: {:?}", inst);
                }
            }
            Err(e) => println!("-- {name}: ERROR {e}"),
        }
    }
}
