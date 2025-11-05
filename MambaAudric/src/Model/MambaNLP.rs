// Program: MambaNLP.rs
// Developer: Audric HARRIS
// Create Date: 5/11/2025
// Update Date: 5/11/2025
// Objective: Implements the full Mamba language model for sentence generation and prediction,
// including embedding, stacked residual blocks, final norm, tied LM head, forward with optional loss,
// and a generation method with sampling options.

use burn::{
    config::Config,
    module::{Module, ModuleHolder, ModuleList},
    nn::{
        embedding::{Embedding, EmbeddingConfig},
        linear::{Linear, LinearConfig},
        loss::CrossEntropyLossConfig,
        norm::{rms_norm::{RMSNorm, RMSNormConfig}, Norm},
    },
    optim::AdamWConfig,
    tensor::{
        backend::Backend,
        ops::{self, BoolTensor, IntTensor},
        Activation, Data, Distribution, Element, Index, Int, Shape, Tensor,
    },
};
use crate::{ModelArgs, ResidualBlock, ResidualBlockConfig, MambaBlockConfig};

#[derive(Config)]
pub struct MambaNLPConfig
{
    vocabSize: usize,
    nLayer: usize,
    dModel: usize,
    dState: usize,
    dConv: usize,
    dInnerFactor: usize,
    dtRank: Option<usize>,
    dropout: f64,
    bias: bool,
    convBias: bool,
    rmsEps: f64,
    lossConfig: CrossEntropyLossConfig,
}

impl Default for MambaNLPConfig
{
    fn default() -> Self
    {
        Self {
            vocabSize: 50_257,
            nLayer: 10,
            dModel: 1024,
            dState: 64,
            dConv: 4,
            dInnerFactor: 2,
            dtRank: None,
            dropout: 0.01,
            bias: false,
            convBias: true,
            rmsEps: 1e-5,
            lossConfig: CrossEntropyLossConfig::new().with_ignore_index(-100i32),
        }
    }
}

impl MambaNLPConfig
{
    pub fn from_model_args(args: &ModelArgs) -> Self
    {
        let dInner = args.dModel * args.dInnerFactor;
        let dtRank = args.dtRank.unwrap_or_else(|| ((dinner as f64 / 16.0).ceil() as usize));
        Self {
            vocabSize: args.vocabSize,
            nLayer: args.nLayer,
            dModel: args.dModel,
            dState: args.dState,
            dConv: args.dConv,
            dInnerFactor: args.dInnerFactor,
            dtRank: Some(dt_rank),
            dropout: args.dropout,
            bias: args.bias,
            convBias: args.convBias,
            rmsEps: 1e-5,
            ..Default::default()
        }
    }
}

#[derive(Module, Debug)]
pub struct MambaNLP<B: Backend>
{
    embedding  : Embedding<B>,
    layers     : ModuleHolder<Vec<ResidualBlock<B>>>,    norm_f: RMSNorm<B>,
    lm_head    : Linear<B>,
    loss_config: CrossEntropyLossConfig,
}

impl MambaNLPConfig
{
    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaNLP<B>
    {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).with_dtype(burn::tensor::Float).init(device);

        let mut layers = Vec::with_capacity(self.n_layer as usize);
        for _ in 0..self.n_layer
        {
            let res_config = ResidualBlockConfig {
                dModel: self.dModel,
                dState: self.dState,
                dConv: self.dConv,
                dInnerFactor: self.dInnerFactor,
                eps: self.rmsEps,
            };
            layers.push(res_config.init(device));
        }
        let layers = ModuleHolder::new(layers);

        let normF = RMSNormConfig::new(self.dModel).with_eps(self.rmsEps).init(device);

        let lmHead = LinearConfig::new(self.dModel, self.vocabSize).with_bias(false).init(device);

        let model = MambaNLP {
            embedding,
            layers,
            normF,
            lmHead,
            lossConfig: self.lossConfig.clone(),
        };

        model.lmHead.weight = model.embedding.weight.clone();

        model
    }

    pub fn from_model_args<B: Backend>(args: &ModelArgs, device: &B::Device) -> MambaNLP<B>
    where
        B: Backend,
    {
        let config = Self::from_model_args(args);
        config.init(device)
    }
}

impl<B: Backend> MambaNLP<B>
{
   pub fn forward(
        &self,
        input_ids: IntTensor<B, 2>,
        targets: Option<IntTensor<B, 2>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 0>>)
   {
        let device = input_ids.device();
        let (batch, seq_len) = input_ids.shape().dims;

        let mut x = self.embedding.forward(input_ids);

        for layer in self.layers.iter()
            x = layer.forward(x);

        x = self.norm_f.forward(x);

        let logits = self.lm_head.forward(x);

        let loss = if let Some(tgts) = targets {
            let flat_logits = logits.clone().view([batch * seq_len, self.lm_head.out_features as usize]);
            let flat_tgts = tgts.flatten(0, 1);
            let loss_fn = self.loss_config.init();
            let ce_loss = loss_fn.forward(flat_logits, flat_tgts);
            Some(ce_loss)
        } else {
            None
        };

        (logits, loss)
    }

    pub fn generate(
        &self,
        mut input_ids: IntTensor<B, 2>,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        device: &B::Device,
    ) -> IntTensor<B, 2>
    {
        let batch = input_ids.shape().dims[0];
        let mut generated = input_ids.clone();

        for _ in 0..max_new_tokens
        {
            let (logits, _) = self.forward(generated.clone().slice([.., generated.shape().dims[1] - 1..generated.shape().dims[1]]), None);
            let logits_last = logits.select(1, 0);

            let scaled_logits = logits_last.clone().div_scalar((temperature as f32).into());

            let next_token = if let (Some(k), Some(p)) = (top_k, top_p) {
                self.sample_top_k_top_p(&scaled_logits, k, p, device)
            } else if top_k.is_some() {
                let k = top_k.unwrap();
                self.sample_top_k(&scaled_logits, k, device)
            } else if top_p.is_some() {
                let p = top_p.unwrap();
                self.sample_top_p(&scaled_logits, p, device)
            } else {
                scaled_logits.argmax(1).unsqueeze(1)
            };

            generated = IntTensor::cat(vec![generated, next_token], 1);
        }

        generated
    }

    fn sample_top_k(
        &self,
        logits: &Tensor<B, 2>,
        k: usize,
        device: &B::Device,
    ) -> IntTensor<B, 2>
    {
        let (vals, indices) = logits.topk(k, -1);
        let mut masked_logits = Tensor::full_like(logits, f32::MIN);
        masked_logits = masked_logits.scatter(-1, &indices.unsqueeze(-1), &vals, Index::Value);
        let probs = masked_logits.softmax(-1);
        probs.multinomial(1, true).int()
    }

    fn sample_top_p(
        &self,
        logits: &Tensor<B, 2>,
        p: f64,
        device: &B::Device,
    ) -> IntTensor<B, 2>
    {
        let sorted_logits = logits.clone().sort_descending(-1, true);
        let sorted_probs = sorted_logits.softmax(-1);
        let cum_probs = sorted_probs.cumsum(-1);
        let mask = cum_probs.gt_scalar(p as f32);
        let mut indices_to_remove = mask.clone();
        indices_to_remove = indices_to_remove.slice([.., 1..]).or(&mask.slice([.., ..indices_to_remove.shape().dims[1] - 1]));
        indices_to_remove = indices_to_remove.slice([.., 0..1]).fill_(false);

        let unsorted_mask = indices_to_remove.scatter(-1, &sorted_logits.indices, &indices_to_remove);
        let probs = logits.softmax(-1).masked_fill(&unsorted_mask, 0.0);
        let renormal = probs.clone().sum_dim(1).unsqueeze(1).recip();
        let probs = probs.mul(&renormal);
        probs.multinomial(1, true).int()
    }

    fn sample_top_k_top_p(
        &self,
        logits: &Tensor<B, 2>,
        k: usize,
        p: f64,
        device: &B::Device,
    ) -> IntTensor<B, 2>
    {
        let truncated = self.sample_top_k(logits, k, device).float();
        let top_k_probs = logits.topk(k, -1).0.softmax(-1);
        let (vals, indices) = logits.topk(k, -1);
        let masked_logits = Tensor::full_like(logits, f32::MIN);
        let masked_logits = masked_logits.scatter(-1, &indices.unsqueeze(-1), &vals);
        self.sample_top_p(&masked_logits, p, device)
    }
}

impl ModelArgs
{
    pub fn build_mamba_nlp<B: Backend>(&self, device: &B::Device) -> MambaNLP<B>
    {
        MambaNLPConfig::from_model_args(self, device)
    }
}
