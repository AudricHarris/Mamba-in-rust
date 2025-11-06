// Program: MambaData.rs
// Developer: Audric HARRIS
// Create Date: 6/11/2025
// Update Date: 6/11/2025
// Objective: Implements the full Mamba language model for sentence generation and prediction,

use burn::{
    data::{dataloader::batcher::Batcher},
    prelude::*,
};

#[derive(Clone, Default)]
pub struct DataBatcher {}

#[derive(Clone, Default)]
pub struct DataBatch<B: Backend> {
    pub text: Tensor<B, 3>,
    pub targers: Tensor<B, 1, Int>,
}


