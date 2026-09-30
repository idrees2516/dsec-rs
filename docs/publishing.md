# Publishing dsec-rs to crates.io

All eight public crates are publish-ready: per-crate metadata
(`description`, `keywords` ≤ 5 × 20 chars, valid `categories` slugs,
crate-level `README.md`), workspace-inherited `version`/`license`/
`repository`, and **versioned path dependencies** in the workspace
`[workspace.dependencies]` table (crates.io resolves those against the
registry at publish time). `dsec-profiling` is `publish = false`
(internal profiling harness).

## Publish order (dependency-resolved)

```
dsec-protocol → dsec-storage → dsec-runtime → dsec-control
              → dsec-sdk → dsec-rl → dsec-firecracker → dsec-bench
```

A crate must exist on crates.io before any crate that depends on it is
published.

## One-time setup

1. Create a publish token at
   <https://crates.io/settings/tokens> (scope: *publish-new*).
2. Add it as the repository secret **`CARGO_REGISTRY_TOKEN`**
   (Settings → Secrets and variables → Actions).
3. (Recommended) Create a `crates-io` GitHub environment
   (Settings → Environments) — the workflow references it, so
   publishes can require manual approval there.

## Publishing a release

Tag and push — `.github/workflows/publish.yml` runs the quality gates
(fmt + clippy + full test suite) and then publishes every crate in
order, and opens a GitHub release with generated notes:

```sh
git tag v0.3.0
git push origin v0.3.0
```

## Publishing manually

```sh
# gate first
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# verify packaging without uploading
cargo publish --dry-run -p dsec-protocol

# then, in order:
cargo login            # paste the token once (stores it locally)
cargo publish -p dsec-protocol
cargo publish -p dsec-storage
cargo publish -p dsec-runtime
cargo publish -p dsec-control
cargo publish -p dsec-sdk
cargo publish -p dsec-rl
cargo publish -p dsec-firecracker
cargo publish -p dsec-bench
```

Notes:

- **Dry-run limitation:** `cargo publish --dry-run` for a downstream
  crate resolves its internal deps against the live crates.io index, so
  it only fully verifies once the upstream crates are uploaded (the
  leaf, `dsec-protocol`, verifies pre-upload — as do all crates after
  the first real publish pass). The workflow publishes strictly in
  dependency order for this reason; quality is gated by the full
  fmt/clippy/test suite before any upload.
- crates.io enforces **one upload per version**: bump the workspace
  `version` (and the `version = "..."` entries on the internal path
  deps) before re-publishing after any change.
- A failed mid-sequence publish is safe to retry from the failed crate
  onward — already-published versions are skipped with an error you
  can ignore (`--allow-dirty` is never needed on a clean tag).
- Ownership transfers / yanking:
  `cargo yank --vers 0.3.0 dsec-runtime` etc.
