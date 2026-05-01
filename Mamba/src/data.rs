// ============================================================
// File: data.rs
// Developer: Audric HARRIS
// Update Date: 30/04/2026
// Objective: Alpaca-cleaned dataset loader + dataset split.
// ============================================================

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use crate::tokenizer::Tokenizer;

// ── Alpaca raw record (instruction / input / output) ────────────────────────

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct AlpacaItem {
    pub instruction: String,
    #[serde(default)]
    pub input:       String,
    pub output:      String,
}

impl AlpacaItem {
    /// Format the record into a single training string.
    ///
    /// With context:
    ///   ### Instruction:\n<instruction>\n\n### Input:\n<input>\n\n### Response:\n<output>
    ///
    /// Without context (`input` is empty):
    ///   ### Instruction:\n<instruction>\n\n### Response:\n<output>
    pub fn to_text(&self) -> String {
        if self.input.trim().is_empty() {
            format!(
                "### Instruction:\n{}\n\n### Response:\n{}",
                self.instruction.trim(),
                self.output.trim(),
            )
        } else {
            format!(
                "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}",
                self.instruction.trim(),
                self.input.trim(),
                self.output.trim(),
            )
        }
    }
}

// ── TextItem (internal training unit) ───────────────────────────────────────

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct TextItem {
    pub text: String,
}

impl From<AlpacaItem> for TextItem {
    fn from(a: AlpacaItem) -> Self {
        Self { text: a.to_text() }
    }
}

// ── Batcher ──────────────────────────────────────────────────────────────────

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

// ── Dataset split ────────────────────────────────────────────────────────────

/// Split a flat `Vec<TextItem>` into (train, valid) slices.
pub fn split_dataset(
    items:      Vec<TextItem>,
    valid_frac: f64,
) -> (Vec<TextItem>, Vec<TextItem>) {
    let n     = items.len();
    let n_val = ((n as f64) * valid_frac).round() as usize;
    let n_val = n_val.max(1);
    let n_tr  = n.saturating_sub(n_val);

    let mut it = items.into_iter();
    let train: Vec<TextItem> = it.by_ref().take(n_tr).collect();
    let valid: Vec<TextItem> = it.collect();
    (train, valid)
}
