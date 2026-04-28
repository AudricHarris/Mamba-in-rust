// ============================================================
// File: training.rs
// Developer: Audric HARRIS
// Update Date: 27/04/2026
// ============================================================

use burn::{
    backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}},
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder,
        metric::LossMetric,
        RegressionOutput,
        TrainOutput,
        TrainStep,
        ValidStep,
    },
};
use burn_dataset::InMemDataset;
use std::path::Path;
use burn::data::dataloader::DataLoaderBuilder;

use crate::data::{MambaBatch, MambaBatcher, TextItem};
use crate::model::{MambaNlp, ModelArgs, MambaNlpConfig};
use crate::tokenizer::Tokenizer;

pub type TrainBackend = Autodiff<Wgpu>;
pub type InferBackend = Wgpu;

pub const SAVE_PATH:    &str = "checkpoints/mamba_model";
pub const ARTIFACT_DIR: &str = "artifacts";


impl<B: AutodiffBackend> TrainStep<MambaBatch<B>, RegressionOutput<B>>
    for MambaNlp<B>
{
    fn step(&self, batch: MambaBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let [batch_size, seq_len] = batch.inputs.dims();
        let (logits, loss) = self.forward(batch.inputs, Some(batch.targets.clone()));
        let loss = loss.expect("targets provided → loss must be Some");

        let vocab   = logits.dims()[2];
        let n       = batch_size * seq_len;
        let output  = logits.reshape([n, vocab]);
        let targets = batch.targets
            .reshape([n])
            .float()
            .unsqueeze_dim::<2>(1);

        let regression_out = RegressionOutput::new(loss.clone(), output, targets);
        TrainOutput::new(self, loss.backward(), regression_out)
    }
}

impl<B: burn::tensor::backend::Backend>
    ValidStep<MambaBatch<B>, RegressionOutput<B>>
    for MambaNlp<B>
{
    fn step(&self, batch: MambaBatch<B>) -> RegressionOutput<B> {
        let [batch_size, seq_len] = batch.inputs.dims();
        let (logits, loss) = self.forward(batch.inputs, Some(batch.targets.clone()));
        let loss = loss.expect("targets provided → loss must be Some");

        let vocab   = logits.dims()[2];
        let n       = batch_size * seq_len;
        let output  = logits.reshape([n, vocab]);
        let targets = batch.targets
            .reshape([n])
            .float()
            .unsqueeze_dim::<2>(1);

        RegressionOutput::new(loss, output, targets)
    }
}


pub fn save_model(model: &MambaNlp<TrainBackend>, path: &str) -> anyhow::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    CompactRecorder::new()
        .record(model.clone().into_record(), path.into())
        .map_err(|e| anyhow::anyhow!("Save failed: {e}"))?;
    println!("Model saved to {path}");
    Ok(())
}

pub fn load_model(
    args:   &ModelArgs,
    path:   &str,
    device: &WgpuDevice,
) -> anyhow::Result<MambaNlp<TrainBackend>> {
    let record = CompactRecorder::new()
        .load(path.into(), device)
        .map_err(|e| anyhow::anyhow!("Load failed: {e}"))?;
    let model = MambaNlpConfig::from_args(args).init(device).load_record(record);
    println!("Model loaded from {path}");
    Ok(model)
}


pub fn run_training_loop(
    model:         MambaNlp<TrainBackend>,
    train_dataset: InMemDataset<TextItem>,
    valid_dataset: InMemDataset<TextItem>,
    tokenizer:     Tokenizer,
    epochs:        usize,
    seq_len:       usize,
    batch_size:    usize,
    learning_rate: f64,
    device:        &WgpuDevice,
) -> anyhow::Result<MambaNlp<TrainBackend>> {
    let batcher_train = MambaBatcher::<TrainBackend>::new(tokenizer.clone(), device.clone(), seq_len);
    let batcher_valid = MambaBatcher::<InferBackend>::new(tokenizer,         device.clone(), seq_len);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .num_workers(4)
        .shuffle(42)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(2)
        .build(valid_dataset);

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .summary()
        .build(
            model,
            AdamConfig::new().init(),
            learning_rate,
        );

    let trained = learner.fit(dataloader_train, dataloader_valid);
    save_model(&trained, SAVE_PATH)?;
    Ok(trained)
}


pub fn train(
    train_dataset: InMemDataset<TextItem>,
    valid_dataset: InMemDataset<TextItem>,
    tokenizer:     Tokenizer,
    args:          &ModelArgs,
    epochs:        usize,
    seq_len:       usize,
    batch_size:    usize,
    learning_rate: f64,
) -> anyhow::Result<MambaNlp<TrainBackend>> {
    let device = WgpuDevice::default();

    let model: MambaNlp<TrainBackend> =
        if Path::new(&format!("{SAVE_PATH}.mpk.gz")).exists() {
            match load_model(args, SAVE_PATH, &device) {
                Ok(m)  => { println!("Resumed from checkpoint."); m }
                Err(e) => {
                    println!("Checkpoint load failed ({e}), starting fresh.");
                    MambaNlpConfig::from_args(args).init(&device)
                }
            }
        } else {
            MambaNlpConfig::from_args(args).init(&device)
        };

    run_training_loop(
        model, train_dataset, valid_dataset, tokenizer,
        epochs, seq_len, batch_size, learning_rate,
        &device,
    )
}

pub fn train_from_checkpoint(
    checkpoint_path: &str,
    train_dataset:   InMemDataset<TextItem>,
    valid_dataset:   InMemDataset<TextItem>,
    tokenizer:       Tokenizer,
    args:            &ModelArgs,
    epochs:          usize,
    seq_len:         usize,
    batch_size:      usize,
    learning_rate:   f64,
) -> anyhow::Result<MambaNlp<TrainBackend>> {
    let device = WgpuDevice::default();
    println!("Loading base model from: {checkpoint_path}");
    let model = load_model(args, checkpoint_path, &device)?;
    println!("Base model loaded. Starting fine-tuning…");

    run_training_loop(
        model, train_dataset, valid_dataset, tokenizer,
        epochs, seq_len, batch_size, learning_rate,
        &device,
    )
}
