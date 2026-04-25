// ============================================================
// Program     : MambaBlock.rs
// Developer   : Audric HARRIS
// Create Date: 5/11/2025
// Update Date: 25/04/2026
// Objective: Handles one block of the huge mamba model 
// ============================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig,
        PaddingConfig1d,
    },
    tensor::{backend::Backend, Tensor, TensorData, activation::softplus, activation::silu},
};

#[derive(Config)]
pub struct MambaBlockConfig {
    pub dim: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
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

        let delta: Tensor<B, 3> = softplus(self.dt_proj.forward(delta_raw), 1.0)
            .clamp(1e-3_f32, 10.0_f32);

        self.selective_scan(input, delta, self.a_log.val(), b_seq, c_seq, self.d_param.val())
    }

    fn selective_scan(
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

        let bar_a: Tensor<B, 4> = (delta4.clone() * a_neg).exp();

        let b4 = b_seq.reshape([batch, seq_len, 1, state_dim]);
        let u4 = u.clone().reshape([batch, seq_len, inner_dim, 1]);
        let bbu: Tensor<B, 4> = delta4 * b4 * u4;

        let mut h: Tensor<B, 3> = Tensor::zeros([batch, inner_dim, state_dim], &device);
        let mut ys: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let a_t   = bar_a.clone().slice([0..batch, t..t+1, 0..inner_dim, 0..state_dim])
                              .reshape([batch, inner_dim, state_dim]);
            let bbu_t = bbu.clone().slice([0..batch, t..t+1, 0..inner_dim, 0..state_dim])
                              .reshape([batch, inner_dim, state_dim]);
            let c_t   = c_seq.clone().slice([0..batch, t..t+1, 0..state_dim])
                              .reshape([batch, 1, state_dim]);

            h = a_t * h + bbu_t;

            let y_t: Tensor<B, 2> = (h.clone() * c_t).sum_dim(2).reshape([batch, inner_dim]);
            ys.push(y_t);
        }

        let y_stacked: Tensor<B, 3> = Tensor::stack(ys, 1);

        let d3 = d.reshape([1, 1, inner_dim]);
        y_stacked + u * d3
    }
}
