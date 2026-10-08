//! Text utilities shared by the transcript, tools, and judges.
//!
//! Ports of upstream `karotte/text.py`, `karotte/truncation.py`, and the
//! metadata sanitizer: surrogate escaping for student-controlled strings,
//! middle-truncation with a tail keep, JSON-length-aware head truncation,
//! and bounded metadata sanitization.

/// Replace unpaired surrogate code points with their backslash-escape form.
///
/// Student-chosen filenames (which appear in error messages) need not be
/// valid UTF-8 upstream; when Python decodes them with `surrogateescape`
/// the lone surrogates would crash `json.dumps`. The Rust port only ever
/// produces valid UTF-8 `String`s, but the function is kept so error text
/// ported from upstream byte streams gets the identical treatment.
pub fn escape_surrogates(text: &str) -> String {
    // U+DC80..U+DCFF are the Python surrogateescape range; escape them.
    text.chars()
        .map(|c| {
            let cp = c as u32;
            if (0xDC80..=0xDCFF).contains(&cp) {
                format!("\\x{:02x}", cp & 0xFF)
            } else {
                c.to_string()
            }
        })
        .collect()
}

/// The marker inserted by [`truncate_middle`].
pub const TRUNCATION_MARKER: &str = "\n[... truncated: exceeded {} characters ...]\n";

/// Characters of the tail kept by [`truncate_middle`] (8192 upstream).
pub const TRUNCATE_TAIL_KEEP: usize = 8192;

/// Truncate keeping the head and (up to) a tail keep, inserting the
/// upstream marker. `text` longer than `max_chars` is shortened so the
/// *result* fits: head keeps `max - tail - marker`, tail keeps
/// `min(8192, max/4)`.
pub fn truncate_middle(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let tail_keep = TRUNCATE_TAIL_KEEP.min(max_chars / 4);
    let marker = marker_for(max_chars);
    // Budget for the marker inside max_chars.
    let marker_len = marker.chars().count();
    let budget = max_chars.saturating_sub(marker_len);
    let head = budget.saturating_sub(tail_keep);
    let head_s: String = text.chars().take(head).collect();
    let tail_s: String = text
        .chars()
        .rev()
        .take(tail_keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head_s}{marker}{tail_s}")
}

/// The exact marker for a given cap.
fn marker_for(max_chars: usize) -> String {
    TRUNCATION_MARKER.replace("{}", &max_chars.to_string())
}

/// Length of `text` once JSON-escaped (upstream `json_encoded_len`, which
/// measures `json.dumps(text, ensure_ascii=False)` byte length).
pub fn json_encoded_len(text: &str) -> usize {
    serde_json::to_string(text).map(|s| s.len()).unwrap_or(0)
}

/// Largest prefix (by chars) of `text` whose JSON encoding fits
/// `max_json_bytes`. Upstream binary-searches with a 10%-shrink estimate;
/// the port does an exact char-boundary scan from the estimate.
pub fn head_within_json_bytes(text: &str, max_json_bytes: usize) -> String {
    let total = json_encoded_len(text);
    if total <= max_json_bytes {
        return text.to_string();
    }
    // Estimate: chars * max / total, then shrink until it fits.
    let char_count = text.chars().count();
    let mut guess = (char_count * max_json_bytes / total).max(1);
    loop {
        let prefix: String = text.chars().take(guess).collect();
        if json_encoded_len(&prefix) <= max_json_bytes {
            // Grow one char while it still fits (handles under-estimates).
            while guess < char_count {
                let longer: String = text.chars().take(guess + 1).collect();
                if json_encoded_len(&longer) <= max_json_bytes {
                    guess += 1;
                    let _ = longer;
                } else {
                    break;
                }
            }
            return prefix;
        }
        guess /= 10;
        if guess == 0 {
            // Even one char may not fit under extreme caps; give the
            // smallest non-empty prefix that does, else empty.
            let one: String = text.chars().take(1).collect();
            return if json_encoded_len(&one) <= max_json_bytes {
                one
            } else {
                String::new()
            };
        }
    }
}

/// Per-value metadata cap (1 MiB of characters, upstream
/// `MAX_METADATA_VALUE_CHARS`).
pub const MAX_METADATA_VALUE_CHARS: usize = 1024 * 1024;

/// Whole-metadata cap (4 MiB of characters, upstream
/// `MAX_METADATA_TOTAL_CHARS`).
pub const MAX_METADATA_TOTAL_CHARS: usize = 4 * 1024 * 1024;

/// Bound each metadata value and the serialized total, escaping surrogates
/// first (upstream `sanitize_metadata`). Values are truncated in place —
/// event consumers must never OOM on student-controlled metadata.
pub fn sanitize_metadata(map: &serde_json::Value) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let mut total: usize = 0;
    match map {
        serde_json::Value::Object(m) => {
            for (k, v) in m {
                let bounded = sanitize_value(v, MAX_METADATA_VALUE_CHARS);
                total += serde_json::to_string(&bounded)
                    .map(|s| s.len())
                    .unwrap_or(0);
                if total > MAX_METADATA_TOTAL_CHARS {
                    break;
                }
                out.insert(k.clone(), bounded);
            }
        }
        other => return sanitize_value(other, MAX_METADATA_TOTAL_CHARS),
    }
    serde_json::Value::Object(out)
}

fn sanitize_value(v: &serde_json::Value, cap: usize) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => {
            let escaped = escape_surrogates(s);
            serde_json::Value::String(truncate_middle(&escaped, cap))
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.iter().map(|x| sanitize_value(x, cap)).collect())
        }
        serde_json::Value::Object(m) => {
            let mut out = serde_json::Map::new();
            for (k, x) in m {
                out.insert(k.clone(), sanitize_value(x, cap));
            }
            serde_json::Value::Object(out)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_surrogates_is_identity_for_plain_text() {
        // Rust strings cannot contain lone surrogates, so the mapping is an
        // identity for all valid inputs — the guard exists for upstream
        // parity where Python may carry surrogateescape-decoded text.
        assert_eq!(escape_surrogates("hello"), "hello");
        assert_eq!(escape_surrogates("日本的道"), "日本的道");
        assert_eq!(
            escape_surrogates("\\x80 already escaped"),
            "\\x80 already escaped"
        );
    }

    #[test]
    fn truncate_middle_short_is_identity() {
        assert_eq!(truncate_middle("short", 100), "short");
    }

    #[test]
    fn truncate_middle_keeps_head_and_tail_with_marker() {
        let text = "HEAD.".to_string() + &"x".repeat(200) + ".TAIL";
        let out = truncate_middle(&text, 120);
        assert!(out.contains("HEAD."), "head kept: {out:?}");
        assert!(out.contains(".TAIL"));
        assert!(out.contains("truncated: exceeded 120 characters"));
        assert!(out.chars().count() <= 120 + TRUNCATION_MARKER.chars().count());
        // small caps keep only a sliver of head — still with the marker
        let tiny = truncate_middle(&text, 60);
        assert!(tiny.contains("truncated: exceeded 60 characters"));
    }

    #[test]
    fn json_encoded_len_matches_serde() {
        let s = "a\"b\\c\n日本";
        assert_eq!(json_encoded_len(s), serde_json::to_string(s).unwrap().len());
    }

    #[test]
    fn head_within_json_bytes_fits_exact() {
        let s = "hello world";
        let whole = json_encoded_len(s);
        assert_eq!(head_within_json_bytes(s, whole), s);
        let cut = head_within_json_bytes(s, whole - 3);
        assert!(json_encoded_len(&cut) <= whole - 3);
        assert!(s.starts_with(&cut));
    }

    #[test]
    fn sanitize_metadata_bounds_values() {
        let mut m = serde_json::Map::new();
        m.insert(
            "k".into(),
            serde_json::Value::String("y".repeat(2 * MAX_METADATA_VALUE_CHARS)),
        );
        let out = sanitize_metadata(&serde_json::Value::Object(m));
        let v = out["k"].as_str().unwrap();
        assert!(v.chars().count() <= MAX_METADATA_VALUE_CHARS + TRUNCATION_MARKER.chars().count());
    }
}
