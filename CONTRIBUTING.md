# Contributing to dsec-rs

Thanks for your interest in improving dsec-rs! This project reimplements the
DSec paper's architecture in Rust; contributions that increase fidelity to
the paper, improve performance, or deepen test coverage are especially
welcome.

## Getting started

1. **Rust 1.85+** (check `rust-toolchain`-free MSRV via `rust-version` in
   `Cargo.toml`). Any recent stable works.
2. Fork & clone, then:

```bash
cargo build --workspace
cargo test --workspace
```

## Development workflow

All of these must pass locally before you open a PR (CI runs the same):

```bash
cargo fmt --all --check               # formatting
cargo clippy --workspace --all-targets -- -D warnings   # zero warnings policy
cargo test --workspace                # 161 tests
```

- **Formatting**: `cargo fmt --all` (config in `rustfmt.toml`, 100-col).
- **Warnings are errors** in CI — fix or justify with a targeted
  `#[allow(...)]` plus a comment explaining why.
- **New features need tests.** Concurrency-sensitive code additionally
  needs a stress/regression test (see
  `resource::tests::concurrent_admission_never_oversubscribes` for the
  pattern).
- **Paper fidelity**: if your change models behavior described in the
  paper, reference the section in a doc comment (`// paper §4.2: ...`).

## Commit style

Conventional commits (`feat:`, `fix:`, `perf:`, `docs:`, `test:`,
`refactor:`, `chore:`). Keep commits focused; `cargo clippy` and `cargo
test` should pass on every commit.

## PR checklist

- [ ] `cargo fmt --all --check` clean
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] `cargo test --workspace` green (run the suite a few times if you
      touched anything concurrent)
- [ ] tests added/updated for behavior changes
- [ ] doc comments updated for API changes
- [ ] `CHANGELOG.md` entry under `[Unreleased]`
- [ ] no new dependencies unless clearly justified (we keep the tree lean)

## Benchmarks

If your change affects a hot path, run the relevant benchmark before/after
and include the numbers in the PR description:

```bash
cargo run --release -p dsec-bench -- <suite> --quick
```

## Reporting bugs / security

See [SECURITY.md](SECURITY.md) for vulnerabilities; regular bugs via GitHub
issues with a reproducer (seed + failing test preferred).
