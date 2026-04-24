// ============================================================
// Program     : ModelArgs.rs
// Developper  : Audric HARRIS
// Create Date : 21/10/2025
// Update Date :  3/11/2025
// Objectif    : Main script to handle model size and hyper params
// ============================================================

use burn::config::Config;

// ------------------------------------------------------------
// Structures
// ------------------------------------------------------------

#[derive(Config, Debug, Clone)]
pub struct ModelArgs {
    d_model: usize,
    d_state: usize,
    d_inner_factor: usize,
    d_conv: usize,
    n_layer: usize,
    vocab_size: usize,
    dt_rank: Option<usize>,
    dropout_rate: f64,
    conv_bias: bool,
    bias: bool,
    use_lm_head: bool,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            d_model: 1024,
            d_state: 64,
            d_inner_factor: 2,
            d_conv: 4,
            n_layer: 10,
            vocab_size: 50257,
            dt_rank: None,
            dropout_rate: 0.01,
            conv_bias: true,
            bias: false,
            use_lm_head: true,
        }
    }
}
