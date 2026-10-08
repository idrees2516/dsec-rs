# Karotte → dsec-karotte

Reference: [preferencemodel/karotte](https://github.com/preferencemodel/karotte)
(MIT / MIT-0), the RL environment framework by Preference Model —
"build robust RL environments to train aligned AI." Sources studied at the
commit pinned in the port's research notes (v3.0.0-era, ~23.4k lines of
Python).

This map links each upstream module to the artifact that re-implements it,
with the same mode legend as `paper-mapping.md`: **P** ported from the
Python original, **L** literal reimplementation of described behavior,
**M** modeled in userspace (the real mechanism is cgroupfs / iptables /
namespaces / a hosted LLM / a Firecracker VMM).

## The design premise

Karotte treats the RL student as a **trained adversary**: whatever raises
its score gets reinforced — reading the answer off disk, killing the
grader, planting symlinks, exhausting the disk, holding RAM in tmpfs or
SysV segments that survive SIGKILL. Three failure classes are never
confused:

| class | meaning | outcome |
|---|---|---|
| task failure | the agent did badly | legitimate zero |
| student misbehavior | the agent attacked the harness | zero, *labeled* (`misbehavior` metadata) |
| infrastructure failure | the grader broke | `ErrorEvent`, run masked |

The other design spine: **a write that succeeded is not evidence of
enforcement**. gVisor accepts cgroup writes and iptables rules and
enforces neither; Karotte detects this and says so via the
`Contract` tri-state (`prevented` / `detected_and_reaped` /
`not_supported`).

## Confinement (confinement.py, cgroups.py, memory_watch.py, process_utils.py)

| upstream concept | artifact | mode |
|---|---|---|
| `Contract` StrEnum (prevented / detected_and_reaped / not_supported) | `confinement::Contract` | **P** |
| `Sandbox` (runc / gvisor / vm; legacy `firecracker` alias) | `confinement::Sandbox::from_env_value` | **P** |
| gVisor excluded from the cgroup path (accepts writes, enforces nothing) | `Sandbox::uses_cgroups` | **P** |
| `HARNESS_RESERVE_BYTES` 1 GiB; 2048 procs; 10^6 files; 0.8 disk | `confinement::ResourceLimits::defaults` | **P** |
| `sandbox_memory_bytes` precedence (env > inherited cgroup > physical) | `confinement::sandbox_memory_bytes` | **P** |
| iptables `-m owner --uid-owner` rules in exact order (blocked ports → loopback → own addrs → internal ranges → allowlist → REJECT; DROP+legacy on gVisor) | `confinement::firewall_plan` | **P** |
| `firewall_canaries` (1.1.1.1:80, metadata, gateway 80/53 from `/proc/net/route` little-endian hex) | `confinement::{firewall_canaries, default_gateway}` | **P** |
| `kill_processes` 30 s deadline, 0.05 s passes, 100 `kill(-1)` rounds, 10 sweeps; clean sweep counts only when the in-cohort kill ran | `confinement::{kill_processes, CohortBackend}` | **M** |
| `UnreapableCohortError` → score 0 | `error::StudentMisbehaviorError::UnreapableCohort` | **P** |
| `unshare --user --net [--map-current-user]` (gVisor omits the mapping) | `confinement::wrap_to_disable_networking` | **P** |
| PID-ns wrapper `--pid --fork --mount-proc --kill-child=SIGKILL --setuid/--setgid` without `--user` | `confinement::pid_namespace_argv` | **P** |
| demotion order: re-enter cwd → fchown stdio → mount ns → IPC ns → oom_score_adj 1000 → setgroups → setgid → setuid | `confinement::demotion_sequence` | **P** |
| `parse_mounts` (v1 controllers minus flags, `name=/mode=/size=` prefixes; octal escapes) | `cgroups::parse_mounts` | **P** |
| v2 delegation: pids into `karotte_harness`, `+memory +pids`, `memory.oom.group=1` | `cgroups::StudentCgroup::delegation_writes` | **M** |
| `memory.max` **paired with `memory.swap.max=0`** (v1: `memory.limit_in_bytes`) | `cgroups::set_memory_limit_writes` | **P** |
| `cgroup.kill=1` atomic subtree kill | `cgroups::kill_all_write` | **P** |
| readback parsing (`max`/`-1`/PAGE_COUNTER_MAX → None; page floor) | `cgroups::{parse_limit_value, page_floor}` | **P** |
| `_weigh`: RSS (`statm`) → PSS (`smaps_rollup`) on the precise pass | `memory_watch::weigh` | **M** |
| tmpfs scratch by `st_blocks`; SysV shm charged to the **creator** (cuid, field 9 of `/proc/sysvipc/shm`) | `memory_watch::{ScratchFile, ShmSegment, parse_sysvipc_shm}` | **P** |
| `disk_backed` longest-prefix, tmpfs/ramfs | `memory_watch::disk_backed` | **P** |
| reap-and-keep-watching (tmpfs/SysV survive SIGKILL) | `memory_watch::MemoryWatch::tick` | **M** |

## Reclaim and custody (reclaim.py, save_submission.py, untrusted_paths.py)

| upstream concept | artifact | mode |
|---|---|---|
| dir-fd-relative-only sweep (`openat`/`fstatat`/`unlinkat`, `O_NOFOLLOW`, NAME_MAX-bounded) | `reclaim::sweep` over `FsView` | **M** |
| virtual fstypes spared; rmdir on the DFS unwind | `reclaim::{VIRTUAL_FSTYPES, FrameAction::RmDir}` | **P** |
| 600 s deadline → `ReclaimError` (misbehavior) | `reclaim::RECLAIM_TIMEOUT_S` | **P** |
| SysV walk bound 2^20, `CAP_IPC_OWNER` bit 15 | `reclaim::{sysv_walk_bound, has_cap_ipc_owner}` | **P** |
| custody copy: symlink/FIFO refusal, 256 MiB/file, 1 GiB total, 10^4 entries, depth 32 | `submission::save_submission` | **M** |
| all-zero chunks skipped as holes; `ftruncate` to logical size | `submission::copy_file` | **P** |
| `_DEST_ERRNOS` (ENAMETOOLONG EEXIST ENOSPC EDQUOT EFBIG) → misbehavior | `submission::map_dest_err` | **P** |
| missing source is *not* misbehavior ("Nothing handed in") | `CustodyReport::missing_source` | **P** |
| `walk_to_parent` refusing intermediate symlinks | `submission::walk_to_parent` | **P** |

## Scoring (task.py, step.py, judges/, evaluation_runner.py)

| upstream concept | artifact | mode |
|---|---|---|
| `Task`/`Step` ABCs (system_prompt, steps, tools, submission_paths union) | `task::{Task, Step}` traits | **P** |
| `Step.score` final: hook → judge; `StudentMisbehaviorError` → score 0 | `task::Step::score` | **P** |
| `misbehavior_scoring` (score 0, `misbehavior` metadata, continue=false) | `schemas::Scoring::misbehavior` | **P** |
| `create_task` dynamic subclass factory | `task::create_task` | **P** |
| `RegexJudge` (join model messages `\n\n`, all patterns must match) | `judges::RegexJudge` (+ a `re`-subset engine) | **P** |
| `RubricJudge`: one LLM call per criterion, YES/NO first-line parse, weighted sum, continue = total **>** threshold | `judges::RubricJudge` over `CompletionClient` | **P** |
| criterion prompt text (dedented) | `RubricJudge::criterion_prompt` | **P** |
| `AnswersContext` / `FileContext` / `TranscriptContext` (tool pairing by id) | `judges::{AnswersContext, FileContext, TranscriptContext}` | **P** |
| `ExecutableJudge`: last arg = output path rewritten to a scratch dir; JSON `{"score", "metadata"}`; continue = score **>=** threshold (default −1) | `judges::ExecutableJudge` | **P** |
| `AndJudge`/`OrJudge` short-circuit; score = last *evaluated*; Unicode tree `✔/❗/⊘`, `├─/└─`, spliced children | `judges::{CompositeJudge, format_judges_tree}` | **P** |
| golden event order: task_started → pre_hook → system → [step_started → … → scoring → step_completed] → task_completed | `runner::Runner::run` (integration-tested) | **P** |
| error path: `ErrorEvent` + `task_completed("error")`; transcript still written durably | `runner::Runner::run` | **P** |

## Agent loop (agents/message_loop.py, builtin_source.py, model_spec.py)

| upstream concept | artifact | mode |
|---|---|---|
| run-wide turn limit (`>` not `>=`), persists across steps | `message_loop::LoopLimits::turn_limit` | **P** |
| time/context limits checked *between* turns (one-turn overrun) | `message_loop::run_step` | **P** |
| `MAX_CONSECUTIVE_EMPTY_TURNS = 3` + the verbatim nudge | `message_loop::{EMPTY_TURN_NUDGE, MAX_CONSECUTIVE_EMPTY_TURNS}` | **P** |
| tool-call buffering; first content block only; images → data URLs | `message_loop::to_chat_content_part` | **P** |
| malformed JSON arguments → `{"_error": ...}` + error result | `message_loop` (test `malformed_arguments_become_error_result`) | **P** |
| "Time remaining: N seconds" / "Context remaining: N" counters | `message_loop::run_step` injection | **P** |
| finish_reason `content_filter` ends the step | `message_loop` | **P** |
| retry ladder: 16 attempts, 60 s cap, Retry-After clamped to 300, 3 auth, 4 unreachable | `message_loop::llm_retry_wait` | **P** |
| effort ladders per family (claude xhigh, gpt-5.x by minor, grok, muse, GLM, Kimi, Qwen) | `model_spec::reasoning_effort_levels` | **P** |
| 128 k output for claude-5/gpt-6 tiers; 64 k default | `model_spec::max_output_tokens` | **P** |
| `supports_sampling_params` negatives; grok temperature 0.7; xai default effort high | `model_spec` | **P** |
| deepseek special-token stripping; xai double-JSON unwrap | `model_spec::repair_{deepseek,xai}_arguments` | **P** |
| provider shaping (anthropic cache_control, openai prompt_cache_key, x-grok-conv-id) | `model_spec::apply_provider_shaping` | **P** |

## Tools and MCP (tools/, mcp_servers/, demoted.py)

| upstream concept | artifact | mode |
|---|---|---|
| bash marker fd 231, `<<exit>>` nonce epilogue, trailing-`&` wrap, `bash -n` syntax check | `tools::{build_bash_command, parse_marker, syntax_check_argv}` | **P** |
| 1 MiB in-memory cap + 8192 tail + the truncation note; 16 000-char final truncation | `tools::{cap_output_memory, truncate_tool_output}` | **P** |
| `view_lines_in_file`: sed `N,Mp;M+1q`, 384 KiB JSON-safe cap, awk line count | `tools::{sed_range_argv, line_count_argv, truncate_content_json_safe}` | **P** |
| `replace_in_file`: 10 MiB cap, projected growth check, hand-rolled unified diff (context 3, hunks merged ≤ 2×context, 50 blocks, 64 KiB, `\ No newline at end of file`) | `tools::unified_diff` | **P** |
| `view_image_file`: 5 MB, dims ≤ 2000, data-URL part | `tools::{validate_image, image_result}` | **P** |
| `/usr/bin/{test,sed,awk,head,tee,base64,bash}` absolute paths | `tools` path constants | **P** |
| MCP JSON-RPC subset: initialize / tools/list / tools/call | `mcp::McpServer::handle` | **P** |
| `register_tools` meta-tool that removes itself | `mcp::McpServer::register_tools` | **P** |
| discovery validation (docstring required, typed params, ToolResult return) | `mcp::validate_tool_descriptor` | **P** |
| resource sampler 0.2 s / 216 000 samples; cgroup v2 cpu.stat/memory.current | `mcp::{ResourceSampler, cpu_percent}` | **M** |
| `_karotte_resource_metrics` injection | `mcp::profile_tool_call` | **P** |

## Transcript and runtime (schemas/, transcript_streaming/, firecracker/, runtime.py)

| upstream concept | artifact | mode |
|---|---|---|
| 14-event union, `type` discriminators, snake_case, MCP wire renames | `schemas::Event` (round-trip tested) | **P** |
| `RunState.apply` (metadata merges, step, score, token sums) | `schemas::RunState::apply` | **P** |
| `EvaluationRunConfig` resolvers (per-step time/context, instruction overrides/extras) | `schemas::EvaluationRunConfig` | **P** |
| stdout renderer glyphs (👤/🗣️/🔧/✅/💥/🏁), bash command special-case, 100 k display cap | `streaming::render_event_stdout` | **P** |
| websocket broadcaster: per-client cursors, replay from 0, completion needs ≥1 client | `streaming::Broadcaster` | **M** |
| backend streamer: positional appends, seq reset, updates on lifecycle+usage events, chunk-free | `streaming::backend_calls` | **M** |
| `Runtime` union + `get_engine` (apple→container, firecracker→docker build) | `runtime::{Runtime, engine}` | **P** |
| `default_runtime` matrix (macOS 26 → apple, kvm → firecracker, else docker) | `runtime::default_runtime` | **P** |
| pinned Firecracker v1.17.0 + Kata 4.0.0 kernel (per-arch SHA-256) | `runtime::{firecracker_artifact, kata_kernel_artifact}` | **P** |
| boot args verbatim + `ip=` TEST-NET-1 arg | `runtime::{BOOT_ARGS, kernel_ip_arg}` | **P** |
| drive order vda base ro / vdb scratch / vdc io / vdd+ mounts (26 cap); io 16 GiB | `runtime::{drive_plan, IO_DRIVE_BYTES}` | **P** |
| vsock CID 3, heartbeat port 52, watchdog 120 s (interval 5, boot grace 180) | `runtime` constants | **P** |
| blocked networks + every host IPv4 rejected except the proxy | `runtime::egress_rules` | **P** |
| DNS fallback 1.1.1.1/8.8.8.8 | `runtime::guest_dns` | **P** |

## Deliberate divergences

1. **Python float formatting** — f-strings print `1.0` where Rust prints
   `1`; the judge-tree formatter keeps Python's shape so metadata trees
   read identically.
2. **serde_json key order** — `Value` object keys iterate sorted
   (BTreeMap); upstream MCP content parts are key-order-insensitive, and
   the struct-level wire types (events, transcript, scoring) keep
   declaration order.
3. **The syscall layer** — namespaces, cgroupfs writes, iptables, and
   the VMM itself are modeled behind small traits; the port builds the
   exact commands and file contents, and in-memory backends verify the
   algorithms. This is the same mode `dsec-agentenv` uses for the pod
   topology and `dsec-firecracker` uses for the fake-VMM harness.
