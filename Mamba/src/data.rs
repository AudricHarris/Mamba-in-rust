// ============================================================
// File: data.rs
// Developer: Audric HARRIS
// Update Date: 27/04/2026
// Objective: Stream HF text dataset into SQLite buffer + dataset split.
// ============================================================

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use crate::tokenizer::Tokenizer;

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct TextItem {
    pub text: String,
}

#[derive(Clone)]
pub struct MambaBatcher<B: Backend> {
    tokenizer: Tokenizer,
    device:    B::Device,
    seq_len:   usize,
}

impl<B: Backend> MambaBatcher<B> {
    pub fn new(tokenizer: Tokenizer, device: B::Device, seq_len: usize) -> Self {
        Self { tokenizer, device, seq_len }
    }
}

#[derive(Clone, Debug)]
pub struct MambaBatch<B: Backend> {
    pub inputs:  Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, TextItem, MambaBatch<B>> for MambaBatcher<B> {
    fn batch(&self, items: Vec<TextItem>, device: &B::Device) -> MambaBatch<B> {
        let batch_size = items.len();
        let mut inputs  = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);

        for item in &items {
            let mut tokens = self.tokenizer.encode(&item.text);
            tokens.truncate(self.seq_len + 1);
            while tokens.len() < self.seq_len + 1 {
                tokens.push(0); // <PAD>
            }
            inputs.extend(tokens[0..self.seq_len].iter().map(|&x| x as i32));
            targets.extend(tokens[1..self.seq_len + 1].iter().map(|&x| x as i32));
        }

        MambaBatch {
            inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(inputs,  [batch_size, self.seq_len]), device,
            ),
            targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(targets, [batch_size, self.seq_len]), device,
            ),
        }
    }
}

pub fn split_dataset(
    dataset:    &burn_dataset::SqliteDataset<TextItem>,
    valid_frac: f64,
) -> (Vec<TextItem>, Vec<TextItem>) {
    let n     = dataset.len();
    let n_val = ((n as f64) * valid_frac).round() as usize;
    let n_val = n_val.max(1);
    let n_tr  = n - n_val;

    let train: Vec<TextItem> = (0..n_tr)
        .filter_map(|i| dataset.get(i))
        .collect();
    let valid: Vec<TextItem> = (n_tr..n)
        .filter_map(|i| dataset.get(i))
        .collect();

    (train, valid)
}
