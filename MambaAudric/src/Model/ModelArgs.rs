// Program     : ModelArgs.rs
// Developper  : Audric HARRIS
// Create Date : 21/10/2025
// Update Date :  3/11/2025
// Objectif    : Main script to handle model size and hyper params


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
#[derive(Config, Debug, Clone)]
pub struct ModelArgs
{
    dModel          : usize,
    dState          : usize,
    factorProjection: usize,
    dConv           : usize,
    nLayer          : usize,
    vocabSize       : usize,

    deltaTMin       : f64,
    deltaTMax       : f64,
    deltaTScale     : f64,
    deltaTInitFloor : f64,
    dropoutRate     : f64,

    convBias        : bool,
    bias            : bool,
    useLmHead       : bool,
}

impl Default for ModelArgs
{
    fn default() -> Self
    {
        return Self {
            dModel          : 1024,
            dState          : 64,
            factorProjection: 2,
            dConv           : 4,
            nLayer          : 10,
            vocabSize       : 50257,

            deltaTMin       : 0.05,
            deltaTMax       : 0.2,
            deltaTScale     : 0.05,
            deltaTInitFloor : 1e-4,
            dropoutRate     : 0.01,

            convBias        : true,
            bias            : false,
            useLmHead       : true,
        };
    }
}

impl ModelArgs
{
    pub fn init<B: Backend>(&self, device: &B::Device) -> Self
    {
        return Self::default();
    }
}

