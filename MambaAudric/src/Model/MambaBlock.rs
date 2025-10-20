use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        linear::{Linear, LinearConfig},
    },
    tensor::{Data, Element, Shape, Tensor, backend::Backend},
};

#[derive(Config)]
pub struct MambaBlockConfig {
    dim: usize,
    d_inner: usize,
    d_state: usize,
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    inProj   : Linear<B>,
    convo    : Conv1d<B>,
    deltaProj: Linear<B>,
    aLog     : Tensor<B, 2>,
    dParam   : Tensor<B, 1>,
    outProj  : Linear<B>,
}

impl MambaBlock
{
    pub fn init<B:Backend>(&self, &B::Device) -> MambaBlock<B>
    {
        
    }

    pub fn forward(&self, input : Tensor) -> Tensor
    {
        
    }

    pub fn ssm(&self, input : Tensor ) -> Tensor
    {

    }

    pub fn SelectiveScan(&self, u, delta, A: Tensor<B, 1>, B: Tensor<B, 1>, C: Tensor<B, 1>, D: Tensor<B, 1>) -> Tensor<B, 1>
    {

    }
}
