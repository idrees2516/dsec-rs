//! Deterministic fake egress router for the Chronus HTTP channel.
//!
//! Real sandboxes reach the network through the node's proxy (Aether
//! tunnels the sandbox's HTTP to the outside world with policy
//! enforcement). In simulation the router serves a fixed, seeded table so
//! tests and RL tasks stay deterministic and offline.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Policy: which URL prefixes a sandbox may reach (paper: egress guarded
/// by eBPF + per-sandbox policy).
#[derive(Debug, Clone)]
pub struct EgressPolicy {
    pub allowed_prefixes: Vec<String>,
}

impl Default for EgressPolicy {
    fn default() -> Self {
        EgressPolicy {
            allowed_prefixes: vec![
                "https://api.internal/".to_string(),
                "https://data.internal/".to_string(),
                "https://registry.dsec/".to_string(),
            ],
        }
    }
}

#[derive(Debug)]
pub struct EgressRouter {
    policy: EgressPolicy,
    routes: HashMap<String, HttpResponse>,
}

impl Default for EgressRouter {
    fn default() -> Self {
        EgressRouter::new(EgressPolicy::default())
    }
}

impl EgressRouter {
    pub fn new(policy: EgressPolicy) -> Self {
        let mut routes = HashMap::new();
        routes.insert(
            "https://api.internal/health".to_string(),
            HttpResponse {
                status: 200,
                body: b"{\"status\":\"ok\"}".to_vec(),
            },
        );
        routes.insert(
            "https://api.internal/version".to_string(),
            HttpResponse {
                status: 200,
                body: b"{\"version\":\"1.0\"}".to_vec(),
            },
        );
        routes.insert(
            "https://data.internal/datasets/list".to_string(),
            HttpResponse {
                status: 200,
                body: b"[\"train\",\"eval\"]".to_vec(),
            },
        );
        routes.insert(
            "https://registry.dsec/v1/images".to_string(),
            HttpResponse {
                status: 200,
                body: b"[\"dsec/agent-base\"]".to_vec(),
            },
        );
        EgressRouter { policy, routes }
    }

    /// Proxies one GET: policy check first, then route lookup.
    pub fn get(&self, url: &str, _headers: &[(String, String)]) -> HttpResponse {
        let allowed = self
            .policy
            .allowed_prefixes
            .iter()
            .any(|p| url.starts_with(p));
        if !allowed {
            return HttpResponse {
                status: 403,
                body: b"blocked by egress policy".to_vec(),
            };
        }
        self.routes.get(url).cloned().unwrap_or(HttpResponse {
            status: 404,
            body: b"not found".to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_and_404() {
        let r = EgressRouter::default();
        let ok = r.get("https://api.internal/health", &[]);
        assert_eq!(ok.status, 200);
        let nf = r.get("https://api.internal/other", &[]);
        assert_eq!(nf.status, 404);
    }

    #[test]
    fn egress_policy_blocks() {
        let r = EgressRouter::default();
        let blocked = r.get("https://evil.example.com/exfil", &[]);
        assert_eq!(blocked.status, 403);
    }
}
