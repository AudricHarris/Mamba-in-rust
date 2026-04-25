// ============================================================
// Program: ResidualBlock.rs
// Developer: Audric HARRIS
// Create Date: 3/11/2025
// Update Date: 24/04/2026
// Objective: Handles the Mamba block with normalization in a residual stack.
// A residual block: x + mixer(norm(x)), with RMSNorm pre-norm.
// ============================================================

use burn::{
    config::Config,
    module::Module,
    nn::RmsNorm,
    nn::RmsNormConfig,
    tensor::{backend::Backend, Tensor},
};
use crate::model::{MambaBlock, MambaBlockConfig};

#[derive(Config)]
pub struct ResidualBlockConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub d_inner_factor: usize,
    pub eps: f64,
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    mixer: MambaBlock<B>,
    norm: RmsNorm<B>,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        let mamba_config = MambaBlockConfig {
            dim: self.d_model,
            d_inner: self.d_model * self.d_inner_factor,
            d_state: self.d_state,
            d_conv: self.d_conv,
        };
        ResidualBlock {
            mixer: mamba_config.init(device),
            norm: RmsNormConfig::new(self.d_model).with_epsilon(self.eps).init(device),
        }
    }
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.norm.forward(x.clone());
        let mixed = self.mixer.forward(normed);
        x + mixed
    }
}
