// Program     : MambaBlock.rs
// Developper  : Audric HARRIS
// Create Date : 21/10/2025
// Update Date :  3/10/2025
// Objectif    : Create a versatile mamba block struct and config struct

// I will be respecting my University writing convention.
// UCC For Classes, LCC For variables, FullCaps for Constants, All Man for indentation
// Functions in burn don't respect my writing convention so I will skip over them

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        linear::{Linear, LinearConfig},
    },
    tensor::{Activation, Data, Element, Shape, Tensor, backend::Backend},
};

#[derive(Config)]
pub struct MambaBlockConfig
{
    dim   : usize,
    dInner: usize,
    dState: usize,
    dConv : usize = 4,
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend>
{
    inProj : Linear<B>,
    conv1d : Conv1d<B>,
    xProj  : Linear<B>,
    dtProj : Linear<B>,
    aLog   : Tensor<B, 2>,
    dParam : Tensor<B, 1>,
    outProj: Linear<B>,
}

impl MambaBlockConfig
{
    pub fn init<B: Backend>(self, device: &B::Device) -> MambaBlock<B>
    {
        let dtRank = (self.dim as f64 / 16.0).ceil() as usize;

        let inProj = LinearConfig::new(self.dim, 2 * self.d_inner).with_bias(false).init(device);

        let conv1d = Conv1dConfig::new(
            self.dInner,
            self.dInner,
            self.dConv,
        ).with_bias(true).with_groups(self.dInner).with_padding(self.dConv - 1).init(device);


        let xProj = LinearConfig::new(self.dInner, dtRank + 2 * self.dState).with_bias(false).init(device);

        let dtProj = LinearConfig::new(dtRank, self.dInner).with_bias(true).init(device);

        let outProj = LinearConfig::new(self.dInner, self.dim).with_bias(false).init(device);

        // Initialize A_log: log(1 to d_state) repeated for each inner dim
        let mut aLogData = vec![0.0f32; self.dInner * self.dState];
        for i in 0..self.dInner
        {
            for j in 0..self.dState
            {
                aLogData[i * self.dState + j] = ((j + 1) as f32).ln();
            }
        }

        let aLogShape = Shape::new([self.dInner as u64, self.dState as u64]);
        let aLogData = Data::new(aLogData, burn::tensor::Kind::Float);
        let aLog = Tensor::from_data(aLogData.convert::<f32>(), aLogShape).to_device(device);

        // D: ones
        let dShape = Shape::new([self.dInner as u64]);
        let dParam = Tensor::ones(dShape, device);

        // Returned MambaBlock
        return MambaBlock{ inProj, conv1d, xProj, dtProj, aLog, dParam, outProj, };
    }
}

impl<B: Backend> MambaBlock<B>
{
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3>
    {
        let (_, seqLen, _) = input.shape().dims;
        let xz = self.inProj.forward(input);
        let (x, z) = xz.split_last_dim(self.inProj.weight.shape().dims[1] / 2);

        let x = x.transpose(1, 2);

        let x = self.conv1d.forward(x);
        let x = x.slice([.., .., 0..seqLen]);
        let x = x.transpose(1, 2);
        
        // pass X through our SSM 
        let x = x.silu();
        let y = self.ssm(x);

        let z = z.silu();
        let y = (y * z).outProj.forward(y);

        // Returns a projected y
        return y;
    }

    pub fn ssm(&self, input: Tensor<B, 3>) -> Tensor<B, 3>
    {
        let dtRank = self.xProj.weight.shape().dims[1] - 2 * self.aLog.shape().dims[1] as usize;
        let xDbl   = self.xProj.forward(input);

        let (deltaRaw, bc) = xDbl.split_last_dim(dtRank);

        let (b, c) = bc.split_last_dim(self.aLog.shape().dims[1] as usize);
        let delta  = self.dtProj.forward(deltaRaw);
        let delta  = delta.softplus();

        return self.selectiveScan(input, delta, self.a_log.clone(), b, c, self.d_param.clone());
    }

    pub fn selectiveScan(
        &self,
        u: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        aLog: Tensor<B, 2>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 3>
    {
        let (batchSize, seqLen, innerDim) = u.shape().dims;
        let stateDim = a_log.shape().dims[1];
        let device = u.device();

        let mut h = Tensor::<B, 3>::zeros(
            Shape::new([batchSize as u64, innerDim as u64, stateDim as u64]),
            &device,
        );

        let mut ys = vec![];

        let a = aLog.exp().neg();

        for t in 0..seqLen
        {
            let uT     = u.select(1, t as i32);
            let deltaT = delta.select(1, t as i32);
            let bT     = b.select(1, t as i32);
            let cT     = c.select(1, t as i32);

            let deltaTExp    = deltaT.unsqueeze_dim(2);
            let aExpanded    = a.unsqueeze_dim(0);
            let atTimesDelta = deltaTExp.matmul(&aExpanded.transpose());
            let barA         = atTimesDelta.exp();

            let one = Tensor::from_floats([1.0f32], &device);
            let numerator = (barA - one.clone().unsqueeze_dim(0).unsqueeze_dim(0).unsqueeze_dim(0));
            let denominator = aExpanded.clone() + one.clone().masked_fill(aExpanded.clone().abs() < Tensor::from_floats([1e-4f32], &device).unsqueeze_dim(0).unsqueeze_dim(0), one);
            let multiplier = numerator / denominator;

            let bTExp = bT.unsqueeze_dim(1);
            let btBar = multiplier * bTExp;

            let uTExp  = uT.unsqueeze_dim(2);
            let deltaH = btBar * uTExp;
            h          = barA * h.clone() + deltaH;

            let cTExp = cT.unsqueeze_dim(1);
            let yInner = (cTExp * h.clone()).sum_dim(2);
            let skip = d.unsqueeze_dim(0) * uT;
            let yT = yInner + skip;

            ys.push(yT);
        }

        let mut y = ys.remove(0);
        
        for yT in ys
            y = Tensor::cat(vec![y, yT], 1);

        return y;
    }
}
