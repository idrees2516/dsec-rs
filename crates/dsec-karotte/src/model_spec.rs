//! The model catalog: per-model parameter derivation.
//!
//! Ports of upstream `karotte/model_spec.py`, `model_catalog.py`, and the
//! request-shaping half of `providers.py`. The prefix/exact tables are
//! ported verbatim so `spec_for("claude-opus-5-5")` derives the same max
//! output tokens, reasoning-effort ladder, sampling support, and request
//! parameters as upstream.

use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Upstream `SPECIAL_TRAINING_MODEL_NAME`.
pub const SPECIAL_TRAINING_MODEL_NAME: &str = "__karotte_special__/training";
/// Upstream `SPECIAL_TRAINING_MODEL_PREFIX` (training checkpoints).
pub const SPECIAL_TRAINING_MODEL_PREFIX: &str = "pt/";
/// Upstream default max output tokens.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 64_000;
/// Grok's pinned temperature (upstream `_GROK_TEMPERATURE`).
pub const GROK_TEMPERATURE: f64 = 0.7;
/// The placeholder key handed to CLI agents when a proxy is in front
/// (upstream `PROXY_PLACEHOLDER_KEY`).
pub const PROXY_PLACEHOLDER_KEY: &str = "model_api_key";

/// Claude families with the adaptive thinking request and 128k output
/// (upstream `_CLAUDE_ADAPTIVE_PREFIXES`).
const CLAUDE_ADAPTIVE_PREFIXES: &[&str] = &[
    "claude-fable",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
];

/// Reasoning effort ladders, lowest first, provider-native names
/// (upstream exact map + ordered prefix rules).
fn reasoning_effort_levels(name: &str) -> &'static [&'static str] {
    const V_LOW_MED_HIGH: &[&str] = &["low", "medium", "high"];
    const V_LOW_MED_HIGH_MAX: &[&str] = &["low", "medium", "high", "max"];
    const V_LOW_MED_HIGH_XHIGH: &[&str] = &["low", "medium", "high", "xhigh"];
    const V_LOW_MED_HIGH_XHIGH_MAX: &[&str] = &["low", "medium", "high", "xhigh", "max"];
    const V_MIN_LOW_MED_HIGH: &[&str] = &["minimal", "low", "medium", "high"];
    const V_GROK: &[&str] = &["low", "medium", "high", "xhigh"];
    const V_MUSE13: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];
    const V_MUSE12: &[&str] = &["minimal", "low", "medium", "high", "xhigh"];
    const V_DEEPSEEK: &[&str] = &["high", "max"];
    const V_GLM: &[&str] = &["low", "high", "max"];
    const V_QWEN: &[&str] = &["low", "medium", "xhigh"];

    // exact models (upstream `_REASONING_EFFORT_MODELS`)
    match name {
        "o1" | "o3" | "gemini-3.1-pro-preview" | "gemini-3.8-flash" | "gemini-3.7-flash" => {
            return V_LOW_MED_HIGH
        }
        "grok-4.7" | "grok-4.6" => return V_GROK,
        "muse-spark-1.2" => return V_MUSE12,
        "muse-spark-1.3" => return V_MUSE13,
        "DeepSeek-V4-Pro-0813" => return V_DEEPSEEK,
        "GLM-5.3" | "glm-5p3" | "glm-5p3-flash" | "Kimi-K3" => return V_GLM,
        "Qwen3.8-2.4T-A95B" => return V_QWEN,
        // the 4-6 Claude generation tops out at max (no xhigh)
        "claude-opus-4-6" | "claude-sonnet-4-6" => return V_LOW_MED_HIGH_MAX,
        _ => {}
    }
    // ordered prefix rules (upstream `_REASONING_EFFORT_PREFIXES`)
    let prefix_rules: &[(&str, &[&str])] = &[
        ("gpt-6", V_LOW_MED_HIGH_XHIGH_MAX),
        ("gpt-5.6", V_LOW_MED_HIGH_XHIGH_MAX),
        ("gpt-5.5", V_LOW_MED_HIGH_XHIGH),
        ("gpt-5.4", V_LOW_MED_HIGH_XHIGH),
        ("gpt-5.3", V_LOW_MED_HIGH),
        ("gpt-5.2", V_LOW_MED_HIGH_XHIGH),
        ("gpt-5.1", V_LOW_MED_HIGH),
        ("gpt-5", V_MIN_LOW_MED_HIGH),
    ];
    for (prefix, levels) in prefix_rules {
        if name.starts_with(prefix) {
            return levels;
        }
    }
    // Claude adaptive families
    for prefix in CLAUDE_ADAPTIVE_PREFIXES {
        if name.starts_with(prefix) {
            // 4-6 handled above (exact), the rest use the xhigh ladder
            return V_LOW_MED_HIGH_XHIGH_MAX;
        }
    }
    &[]
}

/// Max output tokens (upstream exact map + prefix rules).
fn max_output_tokens(name: &str) -> u32 {
    match name {
        "muse-spark-1.3" | "Qwen3.8-2.4T-A95B" => 128_000,
        _ => {
            for prefix in CLAUDE_ADAPTIVE_PREFIXES {
                if name.starts_with(prefix) {
                    return 128_000;
                }
            }
            if name.starts_with("gpt-6") || name.starts_with("gpt-5.6") {
                return 128_000;
            }
            DEFAULT_MAX_OUTPUT_TOKENS
        }
    }
}

/// Models that reject `temperature` (upstream `supports_sampling_params`
/// negatives).
fn supports_sampling_params(name: &str) -> bool {
    let rejects = [
        "gpt-6",
        "claude-fable",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
    ];
    !rejects.iter().any(|p| name.starts_with(p))
}

/// Models needing `reasoning_effort` passed through as an extra OpenAI
/// param (upstream `_EXTRA_ALLOWED_OPENAI_PARAMS` exact set).
fn needs_reasoning_effort_param(name: &str) -> bool {
    const SET: &[&str] = &[
        "muse-spark-1.3",
        "muse-spark-1.2",
        "gemini-3.8-flash",
        "grok-4.7",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "DeepSeek-V4-Pro-0813",
        "GLM-5.3",
        "glm-5p3",
        "glm-5p3-flash",
        "Qwen3.8-2.4T-A95B",
        "Kimi-K3",
    ];
    SET.contains(&name)
}

/// The model spec (upstream `ModelSpec`, frozen pydantic model).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Full model id as configured.
    pub model: String,
    /// Picker display name.
    pub model_display_name: String,
    /// Provider key (`anthropic`, `openai`, `gemini`, `xai`, ...).
    pub provider: String,
    /// Provider display name.
    pub provider_display_name: String,
    /// Model name handed to the completion layer (upstream `litellm_model`).
    pub litellm_model: String,
    /// Max output tokens.
    pub max_output_tokens: u32,
    /// Lowest reasoning effort level.
    pub min_reasoning_effort: Option<String>,
    /// Highest reasoning effort level.
    pub max_reasoning_effort: Option<String>,
    /// Full effort ladder, lowest first.
    pub reasoning_effort_levels: Vec<String>,
    /// Params that request thinking + returned reasoning
    /// (upstream `reasoning_request`).
    pub reasoning_request: BTreeMap<String, Value>,
    /// Whether `temperature` may be sent.
    pub supports_sampling_params: bool,
    /// Pinned temperature (grok: 0.7; others: None).
    pub temperature: Option<f64>,
    /// Provider-agnostic effort param may be forwarded.
    pub extra_reasoning_effort_param: bool,
    /// Tool-call repair dialects (`deepseek`, `xai`).
    pub tool_call_repairs: Vec<&'static str>,
    /// Whether an API key is required.
    pub requires_api_key: bool,
}

/// Provider display names (upstream `_PROVIDER_DISPLAY_NAMES`).
fn provider_display_name(provider: &str) -> String {
    match provider {
        "anthropic" => "Anthropic",
        "openai" => "OpenAI",
        "gemini" => "Google",
        "xai" => "xAI",
        "mistral" => "Mistral",
        "vertex_ai" => "Vertex AI",
        "together_ai" => "Together AI",
        "fireworks_ai" => "Fireworks AI",
        "meta" => "Meta",
        "minimax" => "MiniMax",
        "deepseek" => "DeepSeek",
        "" => "",
        other => other,
    }
    .to_string()
}

/// Upstream `_model_display_name`: last path segment, `_` → space.
fn model_display_name(model: &str) -> String {
    let name = model.rsplit('/').next().unwrap_or(model);
    name.replace('_', " ")
}

/// Whether the model is a special training model.
pub fn is_special_training_model(model: &str) -> bool {
    model == SPECIAL_TRAINING_MODEL_NAME || model.starts_with(SPECIAL_TRAINING_MODEL_PREFIX)
}

/// Derive the spec for a model id (upstream `spec_for`).
pub fn spec_for(model: &str) -> ModelSpec {
    let name = model.rsplit('/').next().unwrap_or(model);
    let (provider, litellm_model) = if model.contains('/') {
        let (prefix, rest) = model.split_once('/').unwrap();
        let litellm = match (prefix, rest) {
            // bare claude ids route to anthropic
            (_, r) if prefix == "anthropic" => format!("anthropic/{r}"),
            // responses-style endpoints
            ("meta", r) if r.starts_with("muse-spark") => format!("meta/responses/{r}"),
            ("openai", r) if r.starts_with("gpt-6") => format!("openai/responses/{r}"),
            _ => model.to_string(),
        };
        (prefix.to_string(), litellm)
    } else if name.starts_with("claude") {
        ("anthropic".to_string(), format!("anthropic/{model}"))
    } else {
        (String::new(), model.to_string())
    };

    let levels = reasoning_effort_levels(name);
    let min_effort = levels.first().map(|s| s.to_string());
    let max_effort = levels.last().map(|s| s.to_string());
    // reasoning_request (upstream `reasoning_request` dict)
    let mut reasoning_request = BTreeMap::new();
    let claude_adaptive = CLAUDE_ADAPTIVE_PREFIXES.iter().any(|p| name.starts_with(p));
    if claude_adaptive {
        reasoning_request.insert(
            "thinking".to_string(),
            json!({"type": "adaptive", "display": "summarized"}),
        );
    } else if name.starts_with("gemini-3") {
        reasoning_request.insert(
            "thinkingConfig".to_string(),
            json!({"includeThoughts": true}),
        );
    } else if name.starts_with("muse-spark")
        || name.starts_with("gpt-6")
        || name.starts_with("gpt-5.6")
    {
        reasoning_request.insert(
            "extra_body".to_string(),
            json!({"reasoning_summary": "detailed"}),
        );
    }

    let temperature = if provider == "xai" {
        Some(GROK_TEMPERATURE)
    } else {
        None
    };

    let mut repairs = Vec::new();
    if model.to_lowercase().contains("deepseek") {
        repairs.push("deepseek");
    }
    if model.to_lowercase().starts_with("xai/") {
        repairs.push("xai");
    }

    let requires_api_key = !model.starts_with("vertex_ai/") && !is_special_training_model(model);

    ModelSpec {
        model: model.to_string(),
        model_display_name: model_display_name(model),
        provider: provider.clone(),
        provider_display_name: provider_display_name(&provider).to_string(),
        litellm_model,
        max_output_tokens: max_output_tokens(name),
        min_reasoning_effort: min_effort,
        max_reasoning_effort: max_effort,
        reasoning_effort_levels: levels.iter().map(|s| s.to_string()).collect(),
        reasoning_request,
        supports_sampling_params: supports_sampling_params(name),
        temperature,
        extra_reasoning_effort_param: needs_reasoning_effort_param(name),
        tool_call_repairs: repairs,
        requires_api_key,
    }
}

impl ModelSpec {
    /// Resolve a configured effort (upstream `reasoning_effort_value`):
    /// `min`/`max` map to the ladder bounds; a literal level passes
    /// through (validated by the caller against the ladder).
    pub fn reasoning_effort_value(&self, level: Option<&str>) -> Option<String> {
        match level {
            None => None,
            Some("min") => self.min_reasoning_effort.clone(),
            Some("max") => self.max_reasoning_effort.clone(),
            Some(other) => Some(other.to_string()),
        }
    }

    /// The default effort (upstream: `high` for xai when available).
    pub fn default_reasoning_effort(&self) -> Option<String> {
        if self.provider == "xai" && self.reasoning_effort_levels.iter().any(|l| l == "high") {
            Some("high".to_string())
        } else {
            None
        }
    }
}

/// Upstream `LlmModel`-level completion parameter assembly
/// (`get_completion_params`): streaming, tools, max tokens, timeouts.
pub fn completion_params(spec: &ModelSpec, tools: &[Value]) -> Value {
    let mut params = json!({
        "stream": true,
        "stream_options": {"include_usage": true},
        "model": spec.litellm_model,
        "max_tokens": spec.max_output_tokens,
        "timeout": 300.0,
    });
    let obj = params.as_object_mut().unwrap();
    if !tools.is_empty() {
        obj.insert("tools".to_string(), Value::Array(tools.to_vec()));
        obj.insert("tool_choice".to_string(), json!("auto"));
    }
    if spec.supports_sampling_params {
        if let Some(t) = spec.temperature {
            obj.insert("temperature".to_string(), json!(t));
        }
    }
    if spec.extra_reasoning_effort_param {
        obj.insert(
            "allowed_openai_params".to_string(),
            json!(["tools", "tool_choice", "reasoning_effort"]),
        );
    } else {
        obj.insert(
            "allowed_openai_params".to_string(),
            json!(["tools", "tool_choice"]),
        );
    }
    params
}

/// Provider request shaping (upstream `providers.py`): cache keys,
/// session headers, reasoning application.
pub fn apply_provider_shaping(spec: &ModelSpec, run_id: &str, params: &mut Value) {
    let obj = match params.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    // Reasoning request first (upstream `apply_reasoning`).
    for (k, v) in &spec.reasoning_request {
        if k == "extra_body" {
            let extra = obj
                .entry("extra_body".to_string())
                .or_insert_with(|| json!({}));
            if let Some(e) = extra.as_object_mut() {
                merge_into(e, v);
            }
        } else {
            obj.insert(k.clone(), v.clone());
        }
    }
    match spec.provider.as_str() {
        "anthropic" => {
            // inline ephemeral cache_control on the last message's last
            // block (upstream applies it on send; here we mark the params)
            obj.insert(
                "prompt_cache_key".to_string(),
                json!({"hint": "anthropic-cache_control", "ttl": "1h"}),
            );
        }
        "openai" => {
            let extra = obj
                .entry("extra_body".to_string())
                .or_insert_with(|| json!({}));
            if let Some(e) = extra.as_object_mut() {
                e.insert("prompt_cache_key".to_string(), json!(run_id));
            }
        }
        "xai" => {
            obj.insert(
                "extra_headers".to_string(),
                json!({"x-grok-conv-id": run_id}),
            );
        }
        "together_ai" => {
            let extra = obj
                .entry("extra_body".to_string())
                .or_insert_with(|| json!({}));
            if let Some(e) = extra.as_object_mut() {
                e.insert("reasoning".to_string(), json!({"enabled": true}));
            }
        }
        _ => {}
    }
}

fn merge_into(target: &mut serde_json::Map<String, Value>, v: &Value) {
    if let Some(m) = v.as_object() {
        for (k, val) in m {
            target.insert(k.clone(), val.clone());
        }
    }
}

/// DeepSeek tool-call repair (upstream providers.py): strip the literal
/// special tokens from argument strings.
pub fn repair_deepseek_arguments(args: &str) -> String {
    const TOKENS: &[&str] = &[
        "<｜tool▁call▁begin｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool▁sep｜>",
        "<｜tool▁outputs▁begin｜>",
        "<｜tool▁outputs▁end｜>",
    ];
    let mut out = args.to_string();
    for t in TOKENS {
        out = out.replace(t, "");
    }
    out
}

/// xAI tool-call repair (upstream): undo double-JSON-encoded string
/// arguments — unwrap a leading/trailing quote pair and the stray
/// prefix pattern.
pub fn repair_xai_arguments(args: &str) -> String {
    let trimmed = args.trim();
    // Fully wrapped: "…"
    if trimmed.len() > 1
        && trimmed.starts_with('"')
        && trimmed.ends_with('"')
        && serde_json::from_str::<Value>(trimmed)
            .map(|v| v.is_string())
            .unwrap_or(false)
    {
        if let Ok(Value::String(inner)) = serde_json::from_str::<Value>(trimmed) {
            // only applied when all string values unwrap
            if serde_json::from_str::<Value>(&inner).is_ok() {
                return inner;
            }
        }
    }
    // Stray quote/separator prefix: ^"?(?:/")?[\s;,]*(/[^\s"]*)$
    let bytes: Vec<char> = trimmed.chars().collect();
    let mut i = 0;
    if i < bytes.len() && bytes[i] == '"' {
        i += 1;
    }
    if i + 1 < bytes.len() && bytes[i] == '/' && bytes[i + 1] == '"' {
        i += 2;
    } else if i < bytes.len() && bytes[i] == '/' {
        i += 1;
    }
    while i < bytes.len() && (bytes[i] == ' ' || bytes[i] == ';' || bytes[i] == ',') {
        i += 1;
    }
    let rest: String = bytes[i..].iter().collect();
    if rest.starts_with('/') && !rest.contains(' ') && !rest.contains('"') {
        return rest;
    }
    args.to_string()
}

/// The model catalog (a representative subset of upstream
/// `CATALOG_MODEL_IDS`, first entry = picker default).
pub fn catalog_model_ids() -> Vec<&'static str> {
    vec![
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-sonnet-5-1",
        "openai/gpt-6",
        "openai/gpt-5.6-luna",
        "openai/o1",
        "gemini/gemini-3.1-pro-preview",
        "vertex_ai/gemini-3.8-flash",
        "xai/grok-4.7",
        "xai/grok-4.6",
        "meta/muse-spark-1.3",
        "meta/muse-spark-1.2",
        "deepseek/DeepSeek-V4-Pro-0813",
        "together_ai/moonshotai/Kimi-K3",
        "fireworks_ai/accounts/fireworks/models/glm-5p3",
        "SPECIAL",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_adaptive_spec() {
        let s = spec_for("claude-opus-5-5");
        assert_eq!(s.provider, "anthropic");
        assert_eq!(s.litellm_model, "anthropic/claude-opus-5-5");
        assert_eq!(s.max_output_tokens, 128_000);
        assert_eq!(
            s.reasoning_effort_levels,
            vec!["low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(s.min_reasoning_effort.as_deref(), Some("low"));
        assert_eq!(s.max_reasoning_effort.as_deref(), Some("max"));
        assert_eq!(
            s.reasoning_request.get("thinking"),
            Some(&json!({"type": "adaptive", "display": "summarized"}))
        );
        assert!(!s.supports_sampling_params);
        assert!(s.requires_api_key);
        assert_eq!(s.model_display_name, "claude-opus-5-5");
        assert_eq!(s.provider_display_name, "Anthropic");
    }

    #[test]
    fn claude_4_6_generation_lacks_xhigh() {
        let s = spec_for("claude-opus-4-6");
        assert_eq!(
            s.reasoning_effort_levels,
            vec!["low", "medium", "high", "max"]
        );
    }

    #[test]
    fn bare_claude_routes_to_anthropic() {
        let s = spec_for("claude-sonnet-5-1");
        assert_eq!(s.provider, "anthropic");
        assert_eq!(s.litellm_model, "anthropic/claude-sonnet-5-1");
    }

    #[test]
    fn gpt6_efforts_and_tokens() {
        let s = spec_for("openai/gpt-6-astra");
        assert_eq!(s.provider, "openai");
        assert_eq!(s.litellm_model, "openai/responses/gpt-6-astra");
        assert_eq!(s.max_output_tokens, 128_000);
        assert_eq!(
            s.reasoning_effort_levels,
            vec!["low", "medium", "high", "xhigh", "max"]
        );
        assert!(!s.supports_sampling_params);
        assert!(s.reasoning_request.contains_key("extra_body"));
        assert!(s.extra_reasoning_effort_param);
    }

    #[test]
    fn gpt5_ladder_degrades_by_minor() {
        assert_eq!(
            spec_for("openai/gpt-5.3-mini").reasoning_effort_levels,
            vec!["low", "medium", "high"]
        );
        assert_eq!(
            spec_for("openai/gpt-5.2").reasoning_effort_levels,
            vec!["low", "medium", "high", "xhigh"]
        );
    }

    #[test]
    fn grok_pins_temperature_and_default_effort() {
        let s = spec_for("xai/grok-4.7");
        assert_eq!(s.provider, "xai");
        assert_eq!(s.temperature, Some(0.7));
        assert_eq!(s.default_reasoning_effort().as_deref(), Some("high"));
        assert_eq!(
            s.reasoning_effort_levels,
            vec!["low", "medium", "high", "xhigh"]
        );
        assert!(s.tool_call_repairs.contains(&"xai"));
    }

    #[test]
    fn deepseek_repair_strips_special_tokens() {
        let dirty = "<｜tool▁call▁begin｜>{\"a\": 1}<｜tool▁call▁end｜>";
        assert_eq!(repair_deepseek_arguments(dirty), "{\"a\": 1}");
    }

    #[test]
    fn xai_repair_unwraps_double_encoded() {
        let wrapped = serde_json::to_string("{\"k\": 1}").unwrap();
        assert_eq!(repair_xai_arguments(&wrapped), "{\"k\": 1}");
        let clean = "{\"k\": 1}";
        assert_eq!(repair_xai_arguments(clean), clean);
    }

    #[test]
    fn effort_resolution_min_max_and_literal() {
        let s = spec_for("claude-opus-5-5");
        assert_eq!(s.reasoning_effort_value(None), None);
        assert_eq!(
            s.reasoning_effort_value(Some("min")).as_deref(),
            Some("low")
        );
        assert_eq!(
            s.reasoning_effort_value(Some("max")).as_deref(),
            Some("max")
        );
        assert_eq!(
            s.reasoning_effort_value(Some("high")).as_deref(),
            Some("high")
        );
    }

    #[test]
    fn special_models_do_not_need_keys() {
        let s = spec_for(SPECIAL_TRAINING_MODEL_NAME);
        assert!(!s.requires_api_key);
        let s2 = spec_for("pt/checkpoint-99");
        assert!(!s2.requires_api_key);
        let s3 = spec_for("vertex_ai/gemini-3.8-flash");
        assert!(!s3.requires_api_key);
    }

    #[test]
    fn completion_params_shape() {
        let s = spec_for("xai/grok-4.7");
        let tools = vec![json!({"type": "function", "function": {"name": "bash"}})];
        let mut p = completion_params(&s, &tools);
        apply_provider_shaping(&s, "run-1", &mut p);
        assert_eq!(p["stream"], json!(true));
        assert_eq!(p["model"], json!("xai/grok-4.7"));
        assert_eq!(p["max_tokens"], json!(64_000));
        assert_eq!(p["tool_choice"], json!("auto"));
        assert_eq!(p["temperature"], json!(0.7));
        assert_eq!(p["extra_headers"]["x-grok-conv-id"], json!("run-1"));
    }

    #[test]
    fn muse_spark_ladder() {
        let s = spec_for("meta/muse-spark-1.3");
        assert_eq!(s.litellm_model, "meta/responses/muse-spark-1.3");
        assert_eq!(
            s.reasoning_effort_levels,
            vec!["minimal", "low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(s.max_output_tokens, 128_000);
    }
}
