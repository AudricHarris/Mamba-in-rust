// ============================================================
// Program: ResidualBlock.rs
// Developer: Audric HARRIS
// Create Date: 3/11/2025
// Update Date: 3/11/2025
// Objective: Handles the Mamba block with normalization in a residual stack.
// A residual block: x + mixer(norm(x)), with RMSNorm pre-norm.
// ============================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        norm::rms_norm::{RMSNorm, RMSNormConfig},
    },
    tensor::{backend::Backend, Tensor},
};
use crate::{MambaBlock, MambaBlockConfig, ModelArgs};

// ------------------------------------------------------------
// Structures
// ------------------------------------------------------------

#[derive(Config)]
pub struct ResidualBlockConfig {
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    d_inner_factor: usize,
    eps: f64,
}

impl Default for ResidualBlockConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            d_state: 64,
            d_conv: 4,
            d_inner_factor: 2,
            eps: 1e-6,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    mixer: MambaBlock<B>,
    norm: RMSNorm<B>,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        let d_inner = self.d_model * self.d_inner_factor;
        let mamba_config = MambaBlockConfig {
            dim: self.d_model,
            d_inner,
            d_state: self.d_state,
            d_conv: self.d_conv,
        };
        let mixer = mamba_config.init(device);
        let norm = RMSNormConfig::new(self.d_model)
            .with_eps(self.eps)
            .init(device);
        ResidualBlock { mixer, norm }
    }

    pub fn from_model_args<B: Backend>(args: &ModelArgs, device: &B::Device) -> ResidualBlock<B> {
        let config = Self {
            d_model: args.d_model,
            d_state: args.d_state,
            d_conv: args.d_conv,
            ..Default::default()
        };
        config.init(device)
    }
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let normalized = self.norm.forward(x);
        let mixed = self.mixer.forward(normalized);
        mixed + residual
    }
}

impl ModelArgs {
    pub fn build_residual_block<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        ResidualBlockConfig::from_model_args(self, device)
    }
}
