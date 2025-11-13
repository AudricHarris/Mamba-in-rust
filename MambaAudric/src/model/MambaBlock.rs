// ============================================================
// Program     : MambaBlock.rs
// Developper  : Audric HARRIS
// Create Date : 21/10/2025
// Update Date :  3/11/2025
// Objectif    : Create a versatile mamba block struct and config struct
// ============================================================

// I decided to stop using university writing convention to take burn writting convention
// Keeps the code consistant with one style of convention

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        linear::{Linear, LinearConfig},
    },
    tensor::{Data, Element, Shape, Tensor, backend::Backend},
};

// ------------------------------------------------------------
// Structures
// ------------------------------------------------------------

#[derive(Config)]
pub struct MambaBlockConfig {
    dim: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
}

impl Default for MambaBlockConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            d_inner: 512,
            d_state: 64,
            d_conv: 4,
        }
    }
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    in_proj: Linear<B>,
    conv_1d: Conv1d<B>,
    x_proj: Linear<B>,
    dt_proj: Linear<B>,
    a_log: Tensor<B, 2>,
    d_param: Tensor<B, 1>,
    out_proj: Linear<B>,
}

impl MambaBlockConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> MambaBlock<B> {
        let dt_rank = (self.dim as f64 / 16.0).ceil() as usize;
        let in_proj = LinearConfig::new(self.dim, 2 * self.d_inner)
            .with_bias(false)
            .init(device);
        let conv_1d = Conv1dConfig::new(self.d_inner, self.d_inner, self.d_conv)
            .with_bias(true)
            .with_groups(self.d_inner)
            .with_padding(self.d_conv - 1)
            .init(device);
        let x_proj =
            LinearConfig::new(self.d_inner, dt_rank + 2 * self.d_state)
                .with_bias(false)
                .init(device);
        let dt_proj = LinearConfig::new(dt_rank, self.d_inner)
            .with_bias(true)
            .init(device);
        let out_proj = LinearConfig::new(self.d_inner, self.dim)
            .with_bias(false)
            .init(device);
        let mut a_log_data = vec![0.0f32; self.d_inner * self.d_state];
        for i in 0..self.d_inner {
            for j in 0..self.d_state {
                a_log_data[i * self.d_state + j] = ((j + 1) as f32).ln();
            }
        }
        let a_log_shape = Shape::new([self.d_inner as u64, self.d_state as u64]);
        let a_log_data = Data::new(a_log_data, burn::tensor::Kind::Float);
        let a_log = Tensor::from_data(a_log_data, a_log_shape).to_device(device);
        let d_shape = Shape::new([self.d_inner as u64]);
        let d_param = Tensor::ones(d_shape, device);
        MambaBlock {
            in_proj,
            conv_1d,
            x_proj,
            dt_proj,
            a_log,
            d_param,
            out_proj,
        }
    }
}

impl<B: Backend> MambaBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (_, seq_len, _) = input.shape().dims;
        let xz = self.in_proj.forward(input);
        let chunk_size = xz.shape().dims[2] / 2;
        let (x, z) = xz.split_last_dim(chunk_size);
        let x = x.transpose(1, 2);
        let x = self.conv_1d.forward(x);
        let x = x.slice([.., .., 0..seq_len]);
        let x = x.transpose(1, 2);
        let x = x.silu();
        let y = self.ssm(x);
        let z = z.silu();
        let y = self.out_proj.forward(y * z);
        y
    }

    pub fn ssm(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let dt_rank = self.x_proj.weight.shape().dims[0] - 2 * self.a_log.shape().dims[1] as usize;
        let x_dbl = self.x_proj.forward(input);
        let (delta_raw, bc) = x_dbl.split_last_dim(dt_rank);
        let (b, c) = bc.split_last_dim(self.a_log.shape().dims[1] as usize);
        let delta = self.dt_proj.forward(delta_raw);
        let delta = delta.softplus();
        self.selective_scan(input, delta, self.a_log.clone(), b, c, self.d_param.clone())
    }

    pub fn selective_scan(
        &self,
        u: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a_log: Tensor<B, 2>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let (batch_size, seq_len, inner_dim) = u.shape().dims;
        let state_dim = a_log.shape().dims[1];
        let device = u.device();
        let mut h = Tensor::zeros(
            Shape::new([batch_size as u64, inner_dim as u64, state_dim as u64]),
            &device,
        );
        let a = a_log.exp().neg();
        let mut ys = vec![];
        let eps = 1e-4f32;
        for t in 0..seq_len {
            let u_t = u.select(1, t as i32);
            let delta_t = delta.select(1, t as i32);
            let b_t = b.select(1, t as i32);
            let c_t = c.select(1, t as i32);
            let delta_t_exp = delta_t.unsqueeze(2);
            let a_expanded = a.unsqueeze(0);
            let delta_a = delta_t_exp * &a_expanded;
            let bar_a = delta_a.exp();
            let numerator = bar_a.sub_scalar(1.0);
            let abs_delta_a = delta_a.abs();
            let small = abs_delta_a.lt_scalar(eps);
            let denom = delta_a.masked_fill(&small, 1.0);
            let sinc = numerator / denom;
            let delta_b = delta_t_exp * b_t.unsqueeze(1);
            let delta_h = sinc * delta_b * u_t.unsqueeze(2);
            h = bar_a * h + delta_h;
            let c_t_exp = c_t.unsqueeze(1);
            let y_inner = (c_t_exp * h).sum_dim(2);
            let skip = d.unsqueeze(0) * u_t;
            let y_t = y_inner + skip;
            ys.push(y_t.unsqueeze(1));
        }
        let y = Tensor::cat(ys, 1);
        y
    }
}
