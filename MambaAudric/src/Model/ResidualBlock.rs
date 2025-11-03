// Program: ResidualBlock.rs
// Developer: Audric HARRIS
// Create Date: 3/11/2025
// Update Date: 3/11/2025
// Objective: Handles the Mamba block with normalization in a residual stack.
// A residual block: x + mixer(norm(x)), with RMSNorm pre-norm.

use burn::{
    config::Config,
    module::Module,
    nn::{
        norm::{rms_norm::{RMSNorm, RMSNormConfig}, Norm},
        linear::LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};
use crate::{ModelArgs, MambaBlock, MambaBlockConfig};

#[derive(Config)]
pub struct ResidualBlockConfig
{
    d_model : usize,
    d_state : usize,
    d_conv  : usize,
    d_inner_factor : usize,
    eps : f64,
}

impl Default for ResidualBlockConfig
{
    fn default() -> Self
    {
        return Self {
            dModel = 1024,
            dState = 64,
            dConv  = 4,
            dInnerFactor = 2,
            eps = 1e-6,
        };
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend>
{
    mixer: MambaBlock<B>,
    norm : RMSNorm<B>,
}

impl ResidualBlockConfig
{
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B>
    {
        let dInner = self.dModel * self.dInnerFactor;
        let mambaConfig = MambaBlockConfig {
            dim: self.dModel,
            dInner,
            dState: self.dState,
            dConv: self.dConv,
        };

        let mixer = mambaConfig.init(device);
        let norm = RMSNormConfig::new(self.dModel).with_eps(self.eps).init(device);

        return ResidualBlock { mixer, norm };
    }

    pub fn fromModelArgs(args: &ModelArgs, device: &B::Device) -> ResidualBlock<B>
    where
        B: Backend,
    {
        let config = Self {
            dModel: args.dModel,
            dState: args.dState,
            dConv: args.donv,
            ..Default::default()
        };

        return config.init(device);
    }
}

impl<B: Backend> ResidualBlock<B>
{
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>
    {
        let residual   = x.clone();
        let normalized = self.norm.forward(x);
        let mixed      = self.mixer.forward(normalized);

        return mixed + residual;
    }
}

impl ModelArgs
{
    pub fn buildResidualBlock<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B>
    {
        return ResidualBlockConfig::fromModelArgs(self, device);
    }
}
