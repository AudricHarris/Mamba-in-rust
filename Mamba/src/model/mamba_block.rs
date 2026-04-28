// ============================================================
// Program     : MambaBlock.rs
// Developer   : Audric HARRIS
// Update Date : 27/04/2026
// Objective   : Mamba SSM block – sequential scan replaced with a
//               parallel associative (prefix-sum) scan.
// ============================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig,
        PaddingConfig1d,
    },
    tensor::{
        backend::Backend,
        activation::{softplus, silu},
        Tensor, TensorData,
    },
};

#[derive(Config)]
pub struct MambaBlockConfig {
    pub dim:          usize,
    pub d_inner:      usize,
    pub d_state:      usize,
    pub d_conv:       usize,
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    in_proj:  Linear<B>,
    conv_1d:  Conv1d<B>,
    x_proj:   Linear<B>,
    dt_proj:  Linear<B>,
    a_log:    burn::module::Param<Tensor<B, 2>>,
    d_param:  burn::module::Param<Tensor<B, 1>>,
    out_proj: Linear<B>,
}

impl MambaBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaBlock<B> {
        let dt_rank = (self.dim as f64 / 16.0).ceil() as usize;

        let in_proj  = LinearConfig::new(self.dim, 2 * self.d_inner).with_bias(false).init(device);
        let conv_1d  = Conv1dConfig::new(self.d_inner, self.d_inner, self.d_conv)
            .with_groups(self.d_inner)
            .with_bias(true)
            .with_padding(PaddingConfig1d::Explicit(self.d_conv - 1))
            .init(device);
        let x_proj   = LinearConfig::new(self.d_inner, dt_rank + 2 * self.d_state).with_bias(false).init(device);
        let dt_proj  = LinearConfig::new(dt_rank, self.d_inner).with_bias(true).init(device);
        let out_proj = LinearConfig::new(self.d_inner, self.dim).with_bias(false).init(device);

        let a_log_data: Vec<f32> = (0..self.d_inner * self.d_state)
            .map(|i| ((i % self.d_state + 1) as f32).ln())
            .collect();
        let a_log = burn::module::Param::from_tensor(
            Tensor::<B, 2>::from_data(
                TensorData::new(a_log_data, [self.d_inner, self.d_state]),
                device,
            )
        );

        let d_param = burn::module::Param::from_tensor(
            Tensor::<B, 1>::ones([self.d_inner], device)
        );

        MambaBlock { in_proj, conv_1d, x_proj, dt_proj, a_log, d_param, out_proj }
    }
}

impl<B: Backend> MambaBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = input.dims();

        let xz: Tensor<B, 3> = self.in_proj.forward(input);
        let [b, s, d_two_inner] = xz.dims();
        let d_inner = d_two_inner / 2;

        let x: Tensor<B, 3> = xz.clone().slice([0..b, 0..s, 0..d_inner]);
        let z: Tensor<B, 3> = xz.slice([0..b, 0..s, d_inner..d_two_inner]);

        let x: Tensor<B, 3> = silu(
            self.conv_1d
                .forward(x.swap_dims(1, 2))
                .slice([0..batch, 0..d_inner, 0..seq_len])
                .swap_dims(1, 2)
        );

        let y: Tensor<B, 3> = self.ssm(x);
        self.out_proj.forward(y * silu(z))
    }

    fn ssm(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, _d] = input.dims();
        let d_state  = self.a_log.val().dims()[1];
        let x_dbl: Tensor<B, 3> = self.x_proj.forward(input.clone());
        let x_dbl_dim = x_dbl.dims()[2];
        let dt_rank = x_dbl_dim - 2 * d_state;

        let delta_raw = x_dbl.clone().slice([0..b, 0..l, 0..dt_rank]);
        let b_seq     = x_dbl.clone().slice([0..b, 0..l, dt_rank..dt_rank + d_state]);
        let c_seq     = x_dbl.slice([0..b, 0..l, dt_rank + d_state..x_dbl_dim]);

        // Softplus ensures Δ > 0; clamp for numerical stability.
        let delta: Tensor<B, 3> = softplus(self.dt_proj.forward(delta_raw), 1.0)
            .clamp(1e-3_f32, 10.0_f32);

        self.selective_scan_parallel(input, delta, self.a_log.val(), b_seq, c_seq, self.d_param.val())
    }

    fn selective_scan_parallel(
        &self,
        u:     Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a_log: Tensor<B, 2>,
        b_seq: Tensor<B, 3>,
        c_seq: Tensor<B, 3>,
        d:     Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, inner_dim] = u.dims();
        let state_dim = a_log.dims()[1];
        let device    = u.device();

        let a_neg = a_log.exp().neg().reshape([1, 1, inner_dim, state_dim]);
        let delta4 = delta.reshape([batch, seq_len, inner_dim, 1]);
        let bar_a: Tensor<B, 4> = (delta4.clone() * a_neg).exp();   // [B, L, D, N]

        let b4 = b_seq.reshape([batch, seq_len, 1, state_dim]);
        let u4 = u.clone().reshape([batch, seq_len, inner_dim, 1]);
        let bar_b: Tensor<B, 4> = delta4 * b4 * u4;                 // [B, L, D, N]

        let mut scan_a: Vec<Tensor<B, 4>> = (0..seq_len)
            .map(|t| bar_a.clone().slice([0..batch, t..t+1, 0..inner_dim, 0..state_dim]))
            .collect();
        let mut scan_b: Vec<Tensor<B, 4>> = (0..seq_len)
            .map(|t| bar_b.clone().slice([0..batch, t..t+1, 0..inner_dim, 0..state_dim]))
            .collect();

        let mut step = 1usize;
        while step < seq_len {
            let mut i = step;
            while i < seq_len {
                let prev = i - step;
                let a_new = scan_a[i].clone() * scan_a[prev].clone();
                let b_new = scan_a[i].clone() * scan_b[prev].clone() + scan_b[i].clone();
                scan_a[i] = a_new;
                scan_b[i] = b_new;
                i += step * 2;
            }
            step *= 2;
        }

        let mut level = seq_len.next_power_of_two() / 2;
        while level >= 1 {
            let mut i = level * 2 - 1;
            while i < seq_len {
                let parent = i - level;
                let a_new = scan_a[i].clone() * scan_a[parent].clone();
                let b_new = scan_a[i].clone() * scan_b[parent].clone() + scan_b[i].clone();
                scan_a[i] = a_new;
                scan_b[i] = b_new;
                i += level * 2;
            }
            if level == 1 { break; }
            level /= 2;
        }

        let mut ys: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let h_t = scan_b[t]
                .clone()
                .reshape([batch, inner_dim, state_dim]);
            let c_t = c_seq
                .clone()
                .slice([0..batch, t..t+1, 0..state_dim])
                .reshape([batch, 1, state_dim]);
            let y_t: Tensor<B, 2> = (h_t * c_t).sum_dim(2).reshape([batch, inner_dim]);
            ys.push(y_t);
        }

        let y_stacked: Tensor<B, 3> = Tensor::stack(ys, 1);
        let d3 = d.reshape([1, 1, inner_dim]);
        y_stacked + u * d3
    }

    pub fn forward_step(
        &self,
        x_tok:      Tensor<B, 2>,
        h_state:    Tensor<B, 3>,
        conv_cache: Tensor<B, 3>,
        device:     &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _d_model] = x_tok.dims();
        let [_b, inner_dim, state_dim] = h_state.dims();
        let d_conv_minus1 = conv_cache.dims()[2];

        let xz = self.in_proj.forward(x_tok);
        let x  = xz.clone().slice([0..batch, 0..inner_dim]);
        let z  = xz.slice([0..batch, inner_dim..2*inner_dim]);

        let x3 = x.clone().unsqueeze_dim::<3>(2);
        let new_cache_full = Tensor::cat(vec![conv_cache.clone(), x3], 2);
        let d_conv = d_conv_minus1 + 1;
        let new_conv_cache = new_cache_full
            .clone()
            .slice([0..batch, 0..inner_dim, 1..d_conv]);

        let weight = self.conv_1d.weight.val();
        let window = new_cache_full;
        let w2 = weight.reshape([inner_dim, d_conv]);
        let x_conv: Tensor<B, 2> = (window * w2.unsqueeze_dim::<3>(0)).sum_dim(2)
            .reshape([batch, inner_dim]);
        let x_conv = if let Some(bias) = self.conv_1d.bias.as_ref() {
            x_conv + bias.val().unsqueeze_dim::<2>(0)
        } else {
            x_conv
        };
        let x_conv = silu(x_conv);

        let dt_rank = {
            let w = self.x_proj.weight.val();
            w.dims()[1] - 2 * state_dim
        };
        let x_dbl = self.x_proj.forward(x_conv.clone());
        let delta_raw = x_dbl.clone().slice([0..batch, 0..dt_rank]);
        let b_tok     = x_dbl.clone().slice([0..batch, dt_rank..dt_rank + state_dim]);
        let c_tok     = x_dbl.slice([0..batch, dt_rank + state_dim..dt_rank + 2 * state_dim]);

        let delta = softplus(self.dt_proj.forward(delta_raw), 1.0)
            .clamp(1e-3_f32, 10.0_f32);

        let a_neg = self.a_log.val().exp().neg();
        let delta3 = delta.clone().unsqueeze_dim::<3>(2);
        let bar_a = (delta3.clone() * a_neg.unsqueeze_dim::<3>(0)).exp();

        let b3 = b_tok.unsqueeze_dim::<3>(1);
        let u3 = x_conv.clone().unsqueeze_dim::<3>(2);
        let bar_b = delta3 * b3 * u3;

        let new_h = bar_a * h_state + bar_b;

        let c3 = c_tok.unsqueeze_dim::<3>(1);
        let y_ssm: Tensor<B, 2> = (new_h.clone() * c3)
            .sum_dim(2)
            .reshape([batch, inner_dim]);
        let d_val = self.d_param.val();
        let y_ssm = y_ssm + x_conv * d_val.unsqueeze_dim::<2>(0);

        let y = self.out_proj.forward(y_ssm * silu(z));

        (y, new_h, new_conv_cache)
    }

    pub fn init_h_state<B2: Backend>(
        batch: usize, inner_dim: usize, state_dim: usize, device: &B2::Device,
    ) -> Tensor<B2, 3> {
        Tensor::zeros([batch, inner_dim, state_dim], device)
    }

    pub fn init_conv_cache<B2: Backend>(
        batch: usize, inner_dim: usize, d_conv: usize, device: &B2::Device,
    ) -> Tensor<B2, 3> {
        Tensor::zeros([batch, inner_dim, d_conv - 1], device)
    }

    pub fn inner_dim(&self) -> usize { self.out_proj.weight.dims()[0] }
    pub fn state_dim(&self) -> usize { self.a_log.val().dims()[1] }
    pub fn d_conv(&self)    -> usize { self.conv_1d.weight.dims()[2] }
}
