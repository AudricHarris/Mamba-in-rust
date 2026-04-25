// ============================================================
// Program: MambaNLP.rs
// Developer: Audric HARRIS
// Create Date: 5/11/2025
// Update Date: 25/04/2026
// Objective: Full Mamba LM with proper top-k sampling and
//            repetition penalty to prevent output loops.
// ============================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig,
        Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        RmsNorm, RmsNormConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};
use crate::model::{ModelArgs, ResidualBlock, ResidualBlockConfig};

#[derive(Config)]
pub struct MambaNlpConfig {
    pub vocab_size: usize,
    pub n_layer: usize,
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub d_inner_factor: usize,
    pub rms_eps: f64,
}

impl MambaNlpConfig {
    pub fn from_args(args: &ModelArgs) -> Self {
        Self {
            vocab_size: args.vocab_size,
            n_layer: args.n_layer,
            d_model: args.d_model,
            d_state: args.d_state,
            d_conv: args.d_conv,
            d_inner_factor: args.d_inner_factor,
            rms_eps: 1e-5,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaNlp<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layer)
            .map(|_| ResidualBlockConfig {
                d_model: self.d_model,
                d_state: self.d_state,
                d_conv: self.d_conv,
                d_inner_factor: self.d_inner_factor,
                eps: self.rms_eps,
            }.init(device))
            .collect();

        MambaNlp {
            embedding,
            layers,
            norm_f: RmsNormConfig::new(self.d_model).with_epsilon(self.rms_eps).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).with_bias(false).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MambaNlp<B: Backend> {
    embedding: Embedding<B>,
    layers: Vec<ResidualBlock<B>>,
    norm_f: RmsNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> MambaNlp<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        targets: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        let [batch, seq_len] = input_ids.dims();
        let mut x = self.embedding.forward(input_ids);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x = self.norm_f.forward(x);
        let logits = self.lm_head.forward(x);

        let loss = targets.map(|tgts| {
            CrossEntropyLossConfig::new()
                .init(&logits.device())
                .forward(
                    logits.clone().reshape([batch * seq_len, self.vocab_size()]),
                    tgts.reshape([batch * seq_len]),
                )
        });
        (logits, loss)
    }

    pub fn vocab_size(&self) -> usize {
        self.lm_head.weight.dims()[1]
    }

    pub fn generate(
        &self,
        mut input_ids: Tensor<B, 2, Int>,
        max_tokens: usize,
        temp: f64,
        top_k: usize,
        rep_penalty: f32,
        _device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        for _ in 0..max_tokens {
            let (logits, _) = self.forward(input_ids.clone(), None);
            let [b, s, v] = logits.dims();

            let last_logit = logits
                .slice([0..b, s - 1..s, 0..v])
                .squeeze::<2>(1);

            let logit_data: Vec<f32> = last_logit
                .clone()
                .into_data()
                .to_vec()
                .expect("logit to_vec failed");

            let ctx_tokens: Vec<i32> = input_ids
                .clone()
                .into_data()
                .to_vec()
                .expect("input_ids to_vec failed");

            let next_token_id = sample_token(
                &logit_data,
                &ctx_tokens,
                v,
                temp as f32,
                top_k,
                rep_penalty,
            );

            let next_token = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(
                    vec![next_token_id as i32],
                    [b, 1],
                ),
                _device,
            );

            input_ids = Tensor::cat(vec![input_ids, next_token], 1);
        }
        input_ids
    }
}

// ------------------------ //
// Pure-CPU sampling helper //
// ------------------------ //

fn sample_token(
    logits: &[f32],
    context: &[i32],
    vocab_size: usize,
    temp: f32,
    top_k: usize,
    rep_penalty: f32,
) -> usize {
    let mut logits: Vec<f32> = logits.to_vec();

    for &tok in context {
        let idx = tok as usize;
        if idx < vocab_size {
            if logits[idx] > 0.0 {
                logits[idx] /= rep_penalty.max(1e-6);
            } else {
                logits[idx] *= rep_penalty.max(1e-6);
            }
        }
    }

    let temp = temp.max(1e-6);
    for l in logits.iter_mut() {
        *l /= temp;
    }

    if top_k > 0 && top_k < vocab_size {
        let mut sorted = logits.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[top_k - 1];
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp() / exp_sum).collect();

    let r: f32 = rand_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }
    probs.iter()
        .enumerate()
        .filter(|&(_, &p)| p > 0.0)
        .last()
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn rand_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new({
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345);
            t ^ 0xdeadbeef_cafebabe
        });
    }
    SEED.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f32) / (u64::MAX as f32)
    })
}
