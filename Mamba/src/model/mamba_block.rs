// ============================================================
// Program     : MambaBlock.rs
// Developer   : Audric HARRIS
// Update Date : 28/04/2026
// Objective   : Mamba SSM block – correct sequential scan.
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
    pub dim:     usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv:  usize,
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

        let in_proj  = LinearConfig::new(self.dim, 2 * self.d_inner)
            .with_bias(false).init(device);
        let conv_1d  = Conv1dConfig::new(self.d_inner, self.d_inner, self.d_conv)
            .with_groups(self.d_inner)
            .with_bias(true)
            .with_padding(PaddingConfig1d::Explicit(self.d_conv - 1))
            .init(device);
        let x_proj   = LinearConfig::new(self.d_inner, dt_rank + 2 * self.d_state)
            .with_bias(false).init(device);

        let dt_proj  = LinearConfig::new(dt_rank, self.d_inner)
            .with_bias(true).init(device);

        let out_proj = LinearConfig::new(self.d_inner, self.dim)
            .with_bias(false).init(device);

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

        let xz        = self.in_proj.forward(input);
        let [b, s, d2] = xz.dims();
        let d_inner    = d2 / 2;

        let x: Tensor<B, 3> = xz.clone().slice([0..b, 0..s, 0..d_inner]);
        let z: Tensor<B, 3> = xz.slice([0..b, 0..s, d_inner..d2]);

        let x: Tensor<B, 3> = silu(
            self.conv_1d
                .forward(x.swap_dims(1, 2))
                .slice([0..batch, 0..d_inner, 0..seq_len])
                .swap_dims(1, 2),
        );

        let y = self.ssm(x);
        self.out_proj.forward(y * silu(z))
    }

    fn ssm(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, _] = input.dims();
        let d_state   = self.a_log.val().dims()[1];

        let x_dbl: Tensor<B, 3> = self.x_proj.forward(input.clone());
        let xd        = x_dbl.dims()[2];
        let dt_rank   = xd - 2 * d_state;

        let delta_raw = x_dbl.clone().slice([0..b, 0..l, 0..dt_rank]);
        let b_seq     = x_dbl.clone().slice([0..b, 0..l, dt_rank..dt_rank + d_state]);
        let c_seq     = x_dbl.slice([0..b, 0..l, dt_rank + d_state..xd]);

        let delta: Tensor<B, 3> = softplus(self.dt_proj.forward(delta_raw), 1.0)
            .clamp(1e-4_f32, 1.0_f32);

        self.selective_scan_sequential(
            input, delta,
            self.a_log.val(),
            b_seq, c_seq,
            self.d_param.val(),
        )
    }

    fn selective_scan_sequential(
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

        let a_neg: Tensor<B, 2> = a_log
            .clamp(0.01_f32, 8.0_f32)
            .exp();

        let mut h: Tensor<B, 3> =
            Tensor::zeros([batch, inner_dim, state_dim], &device);

        let mut ys: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let u_t: Tensor<B, 2> = u
                .clone()
                .slice([0..batch, t..t + 1, 0..inner_dim])
                .reshape([batch, inner_dim]);
            let delta_t: Tensor<B, 2> = delta
                .clone()
                .slice([0..batch, t..t + 1, 0..inner_dim])
                .reshape([batch, inner_dim]);
            let b_t: Tensor<B, 2> = b_seq
                .clone()
                .slice([0..batch, t..t + 1, 0..state_dim])
                .reshape([batch, state_dim]);
            let c_t: Tensor<B, 2> = c_seq
                .clone()
                .slice([0..batch, t..t + 1, 0..state_dim])
                .reshape([batch, state_dim]);

            let delta_3  = delta_t.clone().unsqueeze_dim::<3>(2);
            let a_neg_3  = a_neg.clone().unsqueeze_dim::<3>(0);
            let bar_a: Tensor<B, 3> = (delta_3.clone() * a_neg_3.clone()).neg().exp()
                .clamp(1e-6_f32, 1.0_f32);

            let one_minus_bar_a: Tensor<B, 3> =
                (Tensor::ones_like(&bar_a) - bar_a.clone())
                    / (a_neg_3 + 1e-8_f32);

            let b_3 = b_t.unsqueeze_dim::<3>(1);
            let u_3 = u_t.clone().unsqueeze_dim::<3>(2);
            let bar_b: Tensor<B, 3> = one_minus_bar_a * b_3 * u_3;

            h = bar_a * h + bar_b;

            let c_3 = c_t.unsqueeze_dim::<3>(1);
            let y_t: Tensor<B, 2> = (h.clone() * c_3)
                .sum_dim(2)
                .reshape([batch, inner_dim]);

            let d_2 = d.clone().unsqueeze_dim::<2>(0);
            ys.push(y_t + u_t * d_2);
        }

        Tensor::stack(ys, 1)   // [B, L, D]
    }

    pub fn forward_step(
        &self,
        x_tok:      Tensor<B, 2>,
        h_state:    Tensor<B, 3>,
        conv_cache: Tensor<B, 3>,
        device:     &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _]                = x_tok.dims();
        let [_, inner_dim, state_dim] = h_state.dims();
        let d_conv_minus1             = conv_cache.dims()[2];

        let xz = self.in_proj.forward(x_tok);
        let x  = xz.clone().slice([0..batch, 0..inner_dim]);
        let z  = xz.slice([0..batch, inner_dim..2 * inner_dim]);

        let x3 = x.clone().unsqueeze_dim::<3>(2);
        let new_cache_full = Tensor::cat(vec![conv_cache, x3], 2);
        let d_conv = d_conv_minus1 + 1;
        let new_conv_cache = new_cache_full
            .clone()
            .slice([0..batch, 0..inner_dim, 1..d_conv]);

        let weight = self.conv_1d.weight.val();
        let w2     = weight.reshape([inner_dim, d_conv]);
        let x_conv: Tensor<B, 2> =
            (new_cache_full * w2.unsqueeze_dim::<3>(0))
                .sum_dim(2)
                .reshape([batch, inner_dim]);
        let x_conv = match self.conv_1d.bias.as_ref() {
            Some(b) => x_conv + b.val().unsqueeze_dim::<2>(0),
            None    => x_conv,
        };
        let x_conv = silu(x_conv);

        let dt_rank = {
            let w = self.x_proj.weight.val();
            w.dims()[1] - 2 * state_dim
        };
        let x_dbl     = self.x_proj.forward(x_conv.clone());
        let delta_raw = x_dbl.clone().slice([0..batch, 0..dt_rank]);
        let b_tok     = x_dbl.clone().slice([0..batch, dt_rank..dt_rank + state_dim]);
        let c_tok     = x_dbl.slice([0..batch, dt_rank + state_dim..dt_rank + 2 * state_dim]);

        let delta = softplus(self.dt_proj.forward(delta_raw), 1.0)
            .clamp(1e-4_f32, 1.0_f32);

        let a_neg: Tensor<B, 2> = self.a_log.val()
            .clamp(0.01_f32, 8.0_f32)
            .exp();

        let delta3  = delta.clone().unsqueeze_dim::<3>(2);
        let a_neg3  = a_neg.unsqueeze_dim::<3>(0);
        let bar_a   = (delta3.clone() * a_neg3.clone()).neg().exp()
            .clamp(1e-6_f32, 1.0_f32);

        let one_minus_bar_a: Tensor<B, 3> =
            (Tensor::ones_like(&bar_a) - bar_a.clone()) / (a_neg3 + 1e-8_f32);

        let b3    = b_tok.unsqueeze_dim::<3>(1);
        let u3    = x_conv.clone().unsqueeze_dim::<3>(2);
        let bar_b = one_minus_bar_a * b3 * u3;

        let new_h = bar_a * h_state + bar_b;

        let c3 = c_tok.unsqueeze_dim::<3>(1);
        let y_ssm: Tensor<B, 2> = (new_h.clone() * c3)
            .sum_dim(2)
            .reshape([batch, inner_dim]);
        let y_ssm = y_ssm + x_conv * self.d_param.val().unsqueeze_dim::<2>(0);

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
