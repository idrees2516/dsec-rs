//! LSTM cell (forward pass) with deterministic seeding.
//!
//! PufferLib's buffer stores per-timestep hidden state so recurrent
//! policies can resume mid-episode and reset at boundaries. This cell
//! implements the standard forward equations used by the example policy
//! and the hidden-reset tests:
//!
//! ```text
//! i = σ(W_xi·x + W_hi·h + b_i)      f = σ(W_xf·x + W_hf·h + b_f)
//! o = σ(W_xo·x + W_ho·h + b_o)      g = tanh(W_xg·x + W_hg·h + b_g)
//! c' = f ⊙ c + i ⊙ g                h' = o ⊙ tanh(c')
//! ```

use dsec_protocol::rng::Rng;

/// A single-layer LSTM cell over f32.
#[derive(Debug, Clone)]
pub struct LstmCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Packed weights: [gate][matrix] with gate order i, f, o, g.
    // W_x[gate]: input_size x hidden_size (row-major)
    w_x: Vec<Vec<f32>>,
    // W_h[gate]: hidden_size x hidden_size
    w_h: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
}

const GATES: usize = 4; // i, f, o, g

impl LstmCell {
    /// Xavier-ish scaled init, seeded (deterministic).
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let scale = 1.0 / (input_size.max(1) as f32).sqrt();
        let mut w_x = Vec::with_capacity(GATES);
        let mut w_h = Vec::with_capacity(GATES);
        let mut b = Vec::with_capacity(GATES);
        for _ in 0..GATES {
            let wx: Vec<f32> = (0..input_size * hidden_size)
                .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * scale)
                .collect();
            let wh: Vec<f32> = (0..hidden_size * hidden_size)
                .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * scale)
                .collect();
            let bias = vec![0.0; hidden_size];
            w_x.push(wx);
            w_h.push(wh);
            b.push(bias);
        }
        LstmCell {
            input_size,
            hidden_size,
            w_x,
            w_h,
            b,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Forward step; `h` and `c` are `hidden_size` each (in/out).
    pub fn forward(&self, x: &[f32], h: &mut [f32], c: &mut [f32]) {
        assert_eq!(x.len(), self.input_size, "input size mismatch");
        assert_eq!(h.len(), self.hidden_size);
        assert_eq!(c.len(), self.hidden_size);
        let hid = self.hidden_size;
        let mut gates = vec![0.0f32; GATES * hid];
        for g in 0..GATES {
            let out = &mut gates[g * hid..(g + 1) * hid];
            // W_x·x
            for (col, &xv) in x.iter().enumerate() {
                let row = &self.w_x[g][col * hid..(col + 1) * hid];
                for j in 0..hid {
                    out[j] += xv * row[j];
                }
            }
            // W_h·h
            for (row_i, &hv) in h.iter().enumerate() {
                let row = &self.w_h[g][row_i * hid..(row_i + 1) * hid];
                for j in 0..hid {
                    out[j] += hv * row[j];
                }
            }
            // bias
            for (o, b) in out.iter_mut().zip(&self.b[g]) {
                *o += b;
            }
        }
        let (i_g, f_g, o_g, g_g) = (
            &gates[0..hid],
            &gates[hid..2 * hid],
            &gates[2 * hid..3 * hid],
            &gates[3 * hid..4 * hid],
        );
        let new_c: Vec<f32> = (0..hid)
            .map(|j| {
                let f = Self::sigmoid(f_g[j]);
                let i = Self::sigmoid(i_g[j]);
                let g = g_g[j].tanh();
                f * c[j] + i * g
            })
            .collect();
        let new_h: Vec<f32> = (0..hid)
            .map(|j| Self::sigmoid(o_g[j]) * new_c[j].tanh())
            .collect();
        h.copy_from_slice(&new_h);
        c.copy_from_slice(&new_c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_deterministic() {
        let cell = LstmCell::new(8, 4, 7);
        let mut h1 = vec![0.0; 4];
        let mut c1 = vec![0.0; 4];
        let mut h2 = vec![0.0; 4];
        let mut c2 = vec![0.0; 4];
        let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.4).collect();
        for _ in 0..3 {
            cell.forward(&x, &mut h1, &mut c1);
            cell.forward(&x, &mut h2, &mut c2);
        }
        assert_eq!(h1, h2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn output_bounds() {
        let cell = LstmCell::new(16, 8, 3);
        let mut h = vec![0.0; 8];
        let mut c = vec![0.0; 8];
        let x = vec![0.5; 16];
        for _ in 0..100 {
            cell.forward(&x, &mut h, &mut c);
            for &v in h.iter().chain(c.iter()) {
                assert!(v.abs() <= 1.0, "value {} out of tanh range", v);
            }
        }
    }

    #[test]
    fn different_inputs_diverge() {
        let cell = LstmCell::new(4, 4, 11);
        let mut h = vec![0.0; 4];
        let mut c = vec![0.0; 4];
        cell.forward(&[1.0, 0.0, 0.0, 0.0], &mut h, &mut c);
        let h1 = h.clone();
        cell.forward(&[0.0, 1.0, 0.0, 0.0], &mut h, &mut c);
        assert_ne!(h1, h);
    }
}
