//! IAM: nested projects, hierarchical quotas, tokens and RBAC.
//!
//! Projects form a tree (`root/team-a/sub`); quotas are enforced along
//! the whole ancestor chain, so a child can never exceed what any
//! ancestor granted. Usage is derived from the registry per subtree,
//! which makes enforcement independent of where the sandbox actually
//! runs (the paper's nested-quota semantics).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::model::Quota;
use crate::registry::Registry;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    pub parent: Option<String>,
    pub quota: Quota,
    pub created_epoch_ms: u64,
}

/// Actions guarded by roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    CreateProject,
    CreateSandbox,
    DestroySandbox,
    PauseResume,
    Read,
    Heartbeat,
    Admin,
}

/// Role attached to a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Admin,
    Writer,
    Reader,
}

impl Role {
    pub fn can(&self, action: Action) -> bool {
        match self {
            Role::Admin => true,
            Role::Writer => matches!(
                action,
                Action::CreateSandbox | Action::DestroySandbox | Action::PauseResume | Action::Read
            ),
            Role::Reader => matches!(action, Action::Read),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenInfo {
    pub token: String,
    pub project: String,
    pub role: Role,
    pub created_epoch_ms: u64,
}

#[derive(Debug, Default)]
pub struct Iam {
    projects: RwLock<HashMap<String, Project>>,
    tokens: RwLock<HashMap<String, TokenInfo>>,
    token_seq: AtomicU64,
}

const ROOT: &str = "root";

impl Iam {
    pub fn new() -> Self {
        let iam = Iam::default();
        iam.projects.write().expect("iam poisoned").insert(
            ROOT.to_string(),
            Project {
                name: ROOT.to_string(),
                parent: None,
                quota: Quota::default(),
                created_epoch_ms: 0,
            },
        );
        iam
    }

    /// Root project always exists with unlimited quota.
    pub fn root(&self) -> Project {
        self.projects
            .read()
            .expect("iam poisoned")
            .get(ROOT)
            .cloned()
            .unwrap()
    }

    /// Creates a nested project; parent must exist, name is
    /// `<parent>/<segment>`.
    pub fn create_project(
        &self,
        parent: &str,
        segment: &str,
        quota: Quota,
        epoch_ms: u64,
    ) -> Result<Project> {
        if segment.is_empty() || segment.contains('/') {
            return Err(Error::InvalidArgument(
                "segment must be a single path component".into(),
            ));
        }
        {
            let projects = self.projects.read().expect("iam poisoned");
            if !projects.contains_key(parent) {
                return Err(Error::ParentNotFound(parent.to_string()));
            }
        }
        let name = format!("{}/{}", parent.trim_end_matches('/'), segment);
        let mut projects = self.projects.write().expect("iam poisoned");
        if projects.contains_key(&name) {
            return Err(Error::ProjectExists(name));
        }
        let project = Project {
            name: name.clone(),
            parent: Some(parent.to_string()),
            quota,
            created_epoch_ms: epoch_ms,
        };
        projects.insert(name.clone(), project.clone());
        Ok(project)
    }

    pub fn project(&self, name: &str) -> Result<Project> {
        self.projects
            .read()
            .expect("iam poisoned")
            .get(name)
            .cloned()
            .ok_or_else(|| Error::ProjectNotFound(name.to_string()))
    }

    /// Ancestors from the project itself up to root.
    pub fn ancestors(&self, name: &str) -> Vec<Project> {
        let projects = self.projects.read().expect("iam poisoned");
        let mut chain = Vec::new();
        let mut cur = name.to_string();
        while let Some(p) = projects.get(&cur) {
            chain.push(p.clone());
            match &p.parent {
                Some(parent) => cur = parent.clone(),
                None => break,
            }
        }
        chain
    }

    /// Enforces the quota chain for a prospective sandbox in `project`.
    ///
    /// Walks the ancestor chain under the projects read lock (no
    /// intermediate `Project` clones) and reads usage from the registry's
    /// O(1) index.
    pub fn check_quota(
        &self,
        project: &str,
        req: &crate::model::Resources,
        registry: &Registry,
    ) -> Result<()> {
        let projects = self.projects.read().expect("iam poisoned");
        let mut cur = project;
        while let Some(p) = projects.get(cur) {
            let usage = registry.project_usage(&p.name);
            let q = &p.quota;
            if q.max_sandboxes >= 0 && usage.sandboxes + 1 > q.max_sandboxes {
                return Err(Error::QuotaExceeded {
                    project: p.name.clone(),
                    what: format!("sandboxes {}/{}", usage.sandboxes, q.max_sandboxes),
                });
            }
            if q.max_cpu_millicores >= 0
                && usage.cpu_millicores + req.cpu_millicores > q.max_cpu_millicores
            {
                return Err(Error::QuotaExceeded {
                    project: p.name.clone(),
                    what: format!("cpu {}m/{}m", usage.cpu_millicores, q.max_cpu_millicores),
                });
            }
            if q.max_mem_mib >= 0 && usage.mem_mib + req.mem_mib > q.max_mem_mib {
                return Err(Error::QuotaExceeded {
                    project: p.name.clone(),
                    what: format!("mem {}Mi/{}Mi", usage.mem_mib, q.max_mem_mib),
                });
            }
            match &p.parent {
                Some(parent) => cur = parent,
                None => break,
            }
        }
        Ok(())
    }

    /// Does this project exist? (Existence-only check without a clone.)
    pub fn project_exists(&self, name: &str) -> bool {
        self.projects
            .read()
            .expect("iam poisoned")
            .contains_key(name)
    }

    /// Mints an API token bound to a project and role.
    pub fn create_token(&self, project: &str, role: Role, epoch_ms: u64) -> Result<TokenInfo> {
        self.project(project)?;
        let seq = self.token_seq.fetch_add(1, Ordering::Relaxed);
        let token = format!("dsect_{}_{}", epoch_ms, seq);
        let info = TokenInfo {
            token: token.clone(),
            project: project.to_string(),
            role,
            created_epoch_ms: epoch_ms,
        };
        self.tokens
            .write()
            .expect("iam poisoned")
            .insert(token, info.clone());
        Ok(info)
    }

    /// Validates a bearer token and checks the action is permitted.
    pub fn authorize(&self, token: &str, action: Action, project: &str) -> Result<TokenInfo> {
        let info = self
            .tokens
            .read()
            .expect("iam poisoned")
            .get(token)
            .cloned()
            .ok_or_else(|| Error::Unauthorized("unknown token".into()))?;
        // Tokens can act on their project or any descendant.
        let allowed_project = project == info.project
            || project
                .strip_prefix(info.project.as_str())
                .is_some_and(|rest| rest.starts_with('/'))
            || info.project == ROOT;
        if !allowed_project {
            return Err(Error::Unauthorized(format!(
                "token scope {} excludes {}",
                info.project, project
            )));
        }
        if !info.role.can(action) {
            return Err(Error::Unauthorized(format!(
                "role {:?} cannot perform this action",
                info.role
            )));
        }
        Ok(info)
    }

    pub fn tokens(&self) -> Vec<TokenInfo> {
        let mut v: Vec<TokenInfo> = self
            .tokens
            .read()
            .expect("iam poisoned")
            .values()
            .cloned()
            .collect();
        v.sort_by_key(|a| a.created_epoch_ms);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Resources, SandboxRecord, SandboxSpec, SandboxState};
    use crate::registry::Registry;

    fn iam() -> Iam {
        Iam::new()
    }

    #[test]
    fn nested_project_tree() {
        let iam = iam();
        let a = iam
            .create_project("root", "team-a", Quota::default(), 1)
            .unwrap();
        assert_eq!(a.name, "root/team-a");
        let sub = iam
            .create_project("root/team-a", "sub", Quota::default(), 2)
            .unwrap();
        assert_eq!(sub.name, "root/team-a/sub");
        assert_eq!(iam.ancestors("root/team-a/sub").len(), 3);
        // Duplicate and orphan rejected.
        assert!(matches!(
            iam.create_project("root", "team-a", Quota::default(), 3),
            Err(Error::ProjectExists(_))
        ));
        assert!(matches!(
            iam.create_project("root/ghost", "x", Quota::default(), 3),
            Err(Error::ParentNotFound(_))
        ));
    }

    #[test]
    fn quota_chain_enforced_at_every_level() {
        let iam = iam();
        iam.create_project("root", "limited", Quota::limited(2000, 2048, 2), 1)
            .unwrap();
        iam.create_project("root/limited", "child", Quota::default(), 2)
            .unwrap();
        let registry = Registry::new();
        // Two sandboxes fit.
        for sid in 1..=2 {
            registry.insert_sandbox(SandboxRecord {
                sid,
                spec: SandboxSpec {
                    project: "root/limited/child".into(),
                    ..Default::default()
                },
                node_id: "n".into(),
                state: SandboxState::Ready,
                created_epoch_ms: 0,
                updated_epoch_ms: 0,
            });
        }
        let req = Resources::default();
        // Third exceeds ancestor sandbox quota (parent unlimited, root unlimited).
        assert!(matches!(
            iam.check_quota("root/limited/child", &req, &registry),
            Err(Error::QuotaExceeded { project, .. }) if project == "root/limited"
        ));
    }

    #[test]
    fn quota_rolls_up_from_siblings() {
        let iam = iam();
        iam.create_project("root", "budget", Quota::limited(1000, 1024, -1), 1)
            .unwrap();
        iam.create_project("root/budget", "a", Quota::default(), 2)
            .unwrap();
        iam.create_project("root/budget", "b", Quota::default(), 3)
            .unwrap();
        let registry = Registry::new();
        registry.insert_sandbox(SandboxRecord {
            sid: 1,
            spec: SandboxSpec {
                project: "root/budget/a".into(),
                resources: Resources {
                    cpu_millicores: 600,
                    mem_mib: 600,
                },
                ..Default::default()
            },
            node_id: "n".into(),
            state: SandboxState::Ready,
            created_epoch_ms: 0,
            updated_epoch_ms: 0,
        });
        // Sibling b asking 500 exceeds the shared parent budget of 1000.
        let req = Resources {
            cpu_millicores: 500,
            mem_mib: 100,
        };
        assert!(matches!(
            iam.check_quota("root/budget/b", &req, &registry),
            Err(Error::QuotaExceeded { .. })
        ));
    }

    #[test]
    fn token_scoping_and_roles() {
        let iam = iam();
        iam.create_project("root", "team", Quota::default(), 1)
            .unwrap();
        let writer = iam.create_token("root/team", Role::Writer, 10).unwrap();
        let reader = iam.create_token("root/team", Role::Reader, 11).unwrap();
        let admin = iam.create_token("root", Role::Admin, 12).unwrap();

        assert!(iam
            .authorize(&writer.token, Action::CreateSandbox, "root/team/sub")
            .is_ok());
        assert!(iam
            .authorize(&writer.token, Action::CreateProject, "root/team")
            .is_err());
        assert!(iam
            .authorize(&reader.token, Action::CreateSandbox, "root/team")
            .is_err());
        assert!(iam
            .authorize(&reader.token, Action::Read, "root/team")
            .is_ok());
        // Writer cannot escape its project scope.
        assert!(iam
            .authorize(&writer.token, Action::CreateSandbox, "root/other")
            .is_err());
        // Admin (root scope) can act anywhere.
        assert!(iam
            .authorize(&admin.token, Action::Admin, "root/team")
            .is_ok());
        // Unknown token.
        assert!(iam.authorize("nope", Action::Read, "root").is_err());
    }
}
