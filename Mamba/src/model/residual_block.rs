// ============================================================
// Program: ResidualBlock.rs
// Developer: Audric HARRIS
// Update Date: 27/04/2026
// Objective: Residual block with RMSNorm pre-norm + helper methods
//            that expose the inner MambaBlock for stateful generation.
// ============================================================

use burn::{
    config::Config,
    module::Module,
    nn::{RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, Tensor},
};
use crate::model::{MambaBlock, MambaBlockConfig};

#[derive(Config)]
pub struct ResidualBlockConfig {
    pub d_model:        usize,
    pub d_state:        usize,
    pub d_conv:         usize,
    pub d_inner_factor: usize,
    pub eps:            f64,
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    pub mixer: MambaBlock<B>,
    norm:      RmsNorm<B>,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        let mamba_config = MambaBlockConfig {
            dim:          self.d_model,
            d_inner:      self.d_model * self.d_inner_factor,
            d_state:      self.d_state,
            d_conv:       self.d_conv,
        };
        ResidualBlock {
            mixer: mamba_config.init(device),
            norm:  RmsNormConfig::new(self.d_model).with_epsilon(self.eps).init(device),
        }
    }
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.norm.forward(x.clone());
        let mixed  = self.mixer.forward(normed);
        x + mixed
    }

    pub fn norm_forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x3     = x.clone().unsqueeze_dim::<3>(1);
        let normed = self.norm.forward(x3);
        normed.squeeze(1)
    }

    pub fn mixer_step(
        &self,
        x:          Tensor<B, 2>,
        h_state:    Tensor<B, 3>,
        conv_cache: Tensor<B, 3>,
        device:     &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
        self.mixer.forward_step(x, h_state, conv_cache, device)
    }

    pub fn mixer_inner_dim(&self) -> usize { self.mixer.inner_dim() }
    pub fn mixer_state_dim(&self) -> usize { self.mixer.state_dim() }
    pub fn mixer_d_conv(&self)    -> usize { self.mixer.d_conv() }
}
