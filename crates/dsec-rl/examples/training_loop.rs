//! Agent training loop over real sandboxes (the paper's core workload):
//! a policy interacts with terminal-task environments backed by Chronus
//! sessions; transitions land in the PufferLib-style replay buffer with
//! LSTM hidden-state reset; GAE closes the phase.
//!
//! Run: `cargo run --release --example training_loop -p dsec-rl`

use std::sync::Arc;

use dsec_rl::driver::{Driver, DriverConfig};
use dsec_rl::envpool::EnvPool;
use dsec_rl::sandbox_env::SandboxEnvBuilder;
use dsec_rl::{Action, ReplayBuffer};

fn main() {
    let num_envs = 8;
    let hidden = 32;
    println!(
        "== dsec-rs agent training loop ({} sandbox envs) ==",
        num_envs
    );

    // 1. Environments: real sandboxes on a shared node, one Aether connection.
    let builder = SandboxEnvBuilder::new(42);
    let envs = builder.build_envs(num_envs, 42).unwrap();
    let pool = EnvPool::new(envs).with_workers(4);

    // 2. PufferLib-style buffer with LSTM hidden storage.
    let buffer = ReplayBuffer::new(
        4096,
        num_envs,
        pool.obs_space(),
        pool.action_space(),
        hidden,
    );

    // 3. A deterministic "policy": hash of the observation features.
    let policy = Arc::new(|obs: &[f32], _h: &mut [f32], _c: &mut [f32]| {
        obs.chunks(pool_chunk(obs))
            .map(|o| {
                let key = (o[0] * 977.0 + o[1] * 331.0 + o[15] * 7.0).abs();
                Action::Discrete((key as i64).rem_euclid(6))
            })
            .collect::<Vec<Action>>()
    });

    let mut driver = Driver::new(Box::new(pool), buffer, policy, DriverConfig::default(), 42);

    // 4. Rollout phases.
    for phase in 1..=3 {
        let stats = driver.rollout(200).unwrap();
        driver.finish_phase();
        let violations = driver.check_hidden_invariants();
        println!(
            "phase {}: steps={} episodes={} mean_return={:.3} mean_len={:.1} steps/s={:.0} hidden_reset_violations={}",
            phase,
            stats.steps,
            stats.episodes,
            stats.mean_return,
            stats.mean_len,
            stats.steps_per_sec,
            violations
        );
        // Buffer holds the window; on-policy reset between phases.
        driver.buffer.reset();
    }

    // 5. Sequence sampling with reset-correct h0/c0 (LSTM training shape).
    let mut rng = dsec_protocol::rng::Rng::new(7);
    driver.rollout(100).unwrap();
    driver.finish_phase();
    let batch = driver.buffer.sample_sequences(16, 8, &mut rng);
    println!(
        "sequence batch: seq_len={} batch={} obs/unit={} h0/unit={}",
        batch.seq_len,
        batch.batch,
        batch.obs_size,
        if driver.buffer.hidden_size() > 0 {
            driver.buffer.hidden_size()
        } else {
            0
        }
    );
    println!("== training loop complete ==");
}

fn pool_chunk(obs: &[f32]) -> usize {
    // Observation stride = obs_size (16 for terminal tasks).
    let _ = obs;
    dsec_rl::sandbox_env::TERM_OBS_SIZE
}
