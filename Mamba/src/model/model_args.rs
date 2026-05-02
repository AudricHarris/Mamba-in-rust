// ============================================================
// Program     : ModelArgs.rs
// Developer   : Audric HARRIS
// Update Date : 01/05/2026
// ============================================================

use burn::config::Config;

#[derive(Config, Debug)]
pub struct ModelArgs {
    pub d_model:        usize,
    pub d_state:        usize,
    pub d_inner_factor: usize,
    pub d_conv:         usize,
    pub n_layer:        usize,
    pub vocab_size:     usize,
    pub dt_rank:        Option<usize>,
    pub dropout_rate:   f64,
    pub conv_bias:      bool,
    pub bias:           bool,
    pub use_lm_head:    bool,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            d_model:        256,
            d_state:        16,
            d_inner_factor: 2,
            d_conv:         4,
            n_layer:        6,
            vocab_size:     50257,
            dt_rank:        None,
            dropout_rate:   0.1,
            conv_bias:      true,
            bias:           false,
            use_lm_head:    true,
        }
    }
}
