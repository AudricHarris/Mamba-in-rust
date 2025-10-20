use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        linear::{Linear, LinearConfig},
        loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    },
    tensor::{Data, Element, Shape, Tensor, backend::Backend},
    optim::AdamW,
};


// I made default values for the struct
// In it's current state it would have  136_106_065 params
#[derive(Config)]
pub struct ModelArgs
{
    dModel           : i32 = 1024,
    dState           : i32 = 64,
    FactorProjection : i32 = 2,
    dConv            : i32 = 4,
    nLayer           : i32 = 10,
    vocabSize        : i32 = 50257, // GPT Trained vocabSize

    deltaTMin        : f64 = 0.05,
    deltaTMax        : f64 = 0.2,
    deltaTScale      : f64 = 0.05,
    deltaTInitFloor  : f64 = 1e-4,
    dropoutRate      : f64 = 0.01,

    convBias  : bool = true,
    bias      : bool = false,
    useLmHead : bool = true,

    loss      : CrossEntropyLoss,
    optimizer : AdamW,
}

impl ModelArgs
{
    pub fn init<B: Backend>(&self, &B::Device) -> ModelArgs
    {
        struct test = ModelArgs
    }
}
