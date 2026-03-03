use candle_core::{DType, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct LogitsProcessor {
    rng: StdRng,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    min_p: f64,
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: f64, top_p: f64, top_k: usize) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            temperature: temperature.max(0.0),
            top_p: top_p.clamp(0.0, 1.0),
            top_k,
            min_p: 0.0,
        }
    }

    #[allow(dead_code)]
    pub fn with_min_p(mut self, min_p: f64) -> Self {
        self.min_p = min_p.clamp(0.0, 1.0);
        self
    }

    pub fn sample(&mut self, logits: &Tensor) -> anyhow::Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let mut vals: Vec<f32> = logits.to_vec1()?;

        if self.temperature < 1e-7 {
            return Ok(argmax(&vals) as u32);
        }

        let inv_t = 1.0 / self.temperature as f32;
        for v in vals.iter_mut() {
            *v *= inv_t;
        }

        if self.top_k > 0 && self.top_k < vals.len() {
            self.apply_top_k(&mut vals);
        }

        let mut probs = softmax_inplace(&mut vals);

        if self.min_p > 0.0 {
            self.apply_min_p(&mut probs);
        }

        if self.top_p > 0.0 && self.top_p < 1.0 {
            self.apply_top_p(&mut probs);
        }

        self.sample_categorical(&probs)
    }

    fn apply_top_k(&self, logits: &mut [f32]) {
        let k = self.top_k;
        let mut vals: Vec<f32> = logits.iter().copied().collect();
        vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = vals[k - 1];

        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    fn apply_min_p(&self, probs: &mut Vec<f32>) {
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = max_prob * self.min_p as f32;

        for p in probs.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }

        renormalize(probs);
    }

    fn apply_top_p(&self, probs: &mut Vec<f32>) {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumulative = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative > self.top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }

        let mut keep = vec![false; probs.len()];
        for &(idx, _) in &indexed[..cutoff] {
            keep[idx] = true;
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            }
        }

        renormalize(probs);
    }

    fn sample_categorical(&mut self, probs: &[f32]) -> anyhow::Result<u32> {
        let r: f32 = self.rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return Ok(i as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }
}

fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn softmax_inplace(logits: &mut Vec<f32>) -> &mut Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
    logits
}

fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && (sum - 1.0).abs() > 1e-6 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}
