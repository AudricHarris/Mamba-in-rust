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
