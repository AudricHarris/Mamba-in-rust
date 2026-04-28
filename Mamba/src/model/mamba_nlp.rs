// ============================================================
// Program: MambaNLP.rs
// Developer: Audric HARRIS
// Update Date: 27/04/2026
// Objective: Full Mamba LM.
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
use crate::model::mamba_block::MambaBlock;

#[derive(Config)]
pub struct MambaNlpConfig {
    pub vocab_size:    usize,
    pub n_layer:       usize,
    pub d_model:       usize,
    pub d_state:       usize,
    pub d_conv:        usize,
    pub d_inner_factor: usize,
    pub rms_eps:       f64,
}

impl MambaNlpConfig {
    pub fn from_args(args: &ModelArgs) -> Self {
        Self {
            vocab_size:    args.vocab_size,
            n_layer:       args.n_layer,
            d_model:       args.d_model,
            d_state:       args.d_state,
            d_conv:        args.d_conv,
            d_inner_factor: args.d_inner_factor,
            rms_eps:       1e-5,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaNlp<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layer)
            .map(|_| ResidualBlockConfig {
                d_model:       self.d_model,
                d_state:       self.d_state,
                d_conv:        self.d_conv,
                d_inner_factor: self.d_inner_factor,
                eps:           self.rms_eps,
            }.init(device))
            .collect();

        MambaNlp {
            embedding,
            layers,
            norm_f:  RmsNormConfig::new(self.d_model).with_epsilon(self.rms_eps).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).with_bias(false).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MambaNlp<B: Backend> {
    embedding: Embedding<B>,
    layers:    Vec<ResidualBlock<B>>,
    norm_f:    RmsNorm<B>,
    lm_head:   Linear<B>,
}

impl<B: Backend> MambaNlp<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        targets:   Option<Tensor<B, 2, Int>>,
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
        input_ids:   Tensor<B, 2, Int>,
        max_tokens:  usize,
        temp:        f64,
        top_p:       f64,
        rep_penalty: f32,
        device:      &B::Device,
    ) -> Tensor<B, 2, Int> {
        let [batch, prompt_len] = input_ids.dims();
        let inner_dim  = self.layers[0].mixer_inner_dim();
        let state_dim  = self.layers[0].mixer_state_dim();
        let d_conv     = self.layers[0].mixer_d_conv();

        let mut h_states: Vec<Tensor<B, 3>> = (0..self.layers.len())
            .map(|_| MambaBlock::<B>::init_h_state(batch, inner_dim, state_dim, device))
            .collect();
        let mut conv_caches: Vec<Tensor<B, 3>> = (0..self.layers.len())
            .map(|_| MambaBlock::<B>::init_conv_cache(batch, inner_dim, d_conv, device))
            .collect();

        let (logits, _) = self.forward(input_ids.clone(), None);
        let [b, s, v]   = logits.dims();

        let mut all_ids: Vec<i32> = input_ids
            .clone()
            .into_data()
            .to_vec::<i32>()
            .expect("input_ids to_vec failed");

        let last_logit: Vec<f32> = logits
            .slice([0..b, s - 1..s, 0..v])
            .squeeze::<2>(1)
            .into_data()
            .to_vec::<f32>()
            .expect("logit to_vec failed");

        let next_id = sample_token_top_p(&last_logit, &all_ids, v, temp as f32, top_p as f32, rep_penalty);
        all_ids.push(next_id as i32);

        for t in 0..prompt_len {
            let tok_id: i32 = input_ids
                .clone()
                .slice([0..batch, t..t+1])
                .into_data()
                .to_vec::<i32>()
                .unwrap()[0];
            let tok_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![tok_id], [1, 1]),
                device,
            );
            let emb: Tensor<B, 2> = self.embedding.forward(tok_tensor).squeeze(1); // [1, D]
            let mut x = emb;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let norm_x = layer.norm_forward(x.clone());
                let (y, new_h, new_cache) = layer.mixer_step(
                    norm_x,
                    h_states[layer_idx].clone(),
                    conv_caches[layer_idx].clone(),
                    device,
                );
                x = x + y;
                h_states[layer_idx] = new_h;
                conv_caches[layer_idx] = new_cache;
            }
        }

        for _ in 1..max_tokens {
            let last_id = *all_ids.last().unwrap();
            let tok_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![last_id], [1, 1]),
                device,
            );
            let emb: Tensor<B, 2> = self.embedding.forward(tok_tensor).squeeze(1); // [1, D]
            let mut x = emb;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let norm_x = layer.norm_forward(x.clone());
                let (y, new_h, new_cache) = layer.mixer_step(
                    norm_x,
                    h_states[layer_idx].clone(),
                    conv_caches[layer_idx].clone(),
                    device,
                );
                x = x + y;
                h_states[layer_idx] = new_h;
                conv_caches[layer_idx] = new_cache;
            }

            let x3 = x.clone().unsqueeze_dim::<3>(1);                 // [1, 1, D]
            let normed = self.norm_f.forward(x3);
            let logit_2d = self.lm_head.forward(normed).squeeze::<2>(1); // [1, V]
            let logit_vec: Vec<f32> = logit_2d
                .into_data()
                .to_vec::<f32>()
                .expect("logit to_vec");

            let vocab_size = self.vocab_size();
            let next = sample_token_top_p(
                &logit_vec, &all_ids, vocab_size,
                temp as f32, top_p as f32, rep_penalty,
            );
            all_ids.push(next as i32);
        }

        // Return only the newly generated tokens (after the prompt).
        let new_tokens: Vec<i32> = all_ids[prompt_len..].to_vec();
        let new_len = new_tokens.len();
        Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(new_tokens, [1, new_len]),
            device,
        )
    }
}

fn sample_token_top_p(
    logits:      &[f32],
    context:     &[i32],
    vocab_size:  usize,
    temp:        f32,
    top_p:       f32,
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
    for l in logits.iter_mut() { *l /= temp; }

    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max_l).exp() / exp_sum))
        .collect();

    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_p = top_p.clamp(0.0, 1.0);
    let mut cumsum = 0.0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for (idx, p) in &probs {
        nucleus.push((*idx, *p));
        cumsum += p;
        if cumsum >= top_p { break; }
    }

    let nucleus_sum: f32 = nucleus.iter().map(|&(_, p)| p).sum();
    let r = rand_f32() * nucleus_sum;
    let mut cumsum = 0.0f32;
    for (idx, p) in &nucleus {
        cumsum += p;
        if r <= cumsum { return *idx; }
    }
    nucleus.last().map(|&(i, _)| i).unwrap_or(0)
}

fn rand_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new({
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345)
                ^ 0xdeadbeef_cafebabe
        });
    }
    SEED.with(|s| {
        let mut x = s.get();
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        s.set(x);
        (x as f32) / (u64::MAX as f32)
    })
}
