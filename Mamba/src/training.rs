use burn::{
    backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}},
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor, TensorData},
};
use std::io::Write;
use std::path::Path;

use crate::model::MambaNlp;
use crate::model::ModelArgs;
use crate::model::MambaNlpConfig;
use crate::tokenizer::Tokenizer;

pub type TrainBackend = Autodiff<Wgpu>;

pub const SAVE_PATH: &str = "checkpoints/mamba_model";

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
    args: &ModelArgs,
    path: &str,
    device: &WgpuDevice,
) -> anyhow::Result<MambaNlp<TrainBackend>> {
    let record = CompactRecorder::new()
        .load(path.into(), device)
        .map_err(|e| anyhow::anyhow!("Load failed: {e}"))?;
    let model = MambaNlpConfig::from_args(args).init(device).load_record(record);
    println!("Model loaded from {path}");
    Ok(model)
}

pub fn train(
    texts: &[String],
    tokenizer: &Tokenizer,
    args: &ModelArgs,
    epochs: usize,
    seq_len: usize,
    batch_size: usize,
    max_tokens: Option<usize>,
    learning_rate: f64,
) -> MambaNlp<TrainBackend> {
    let device = WgpuDevice::default();

    let mut model: MambaNlp<TrainBackend> = if Path::new(&format!("{SAVE_PATH}.mpk.gz")).exists() {
        match load_model(args, SAVE_PATH, &device) {
            Ok(m) => { println!("Resumed from checkpoint."); m }
            Err(e) => {
                println!("Checkpoint load failed ({e}), starting fresh.");
                MambaNlpConfig::from_args(args).init(&device)
            }
        }
    } else {
        MambaNlpConfig::from_args(args).init(&device)
    };

    let mut optim = AdamConfig::new()
        .with_epsilon(1e-7)
        .init::<TrainBackend, MambaNlp<TrainBackend>>();

    let mut all_tokens: Vec<usize> = Vec::new();
    for text in texts {
        let mut enc = tokenizer.encode(text);
        enc.push(2);
        all_tokens.extend(enc);
    }
    if let Some(limit) = max_tokens {
        all_tokens.truncate(limit);
    }

    let chunks: Vec<&[usize]> = all_tokens
        .windows(seq_len + 1)
        .step_by(seq_len)
        .collect();
    let total_steps_per_epoch = (chunks.len() as f32 / batch_size as f32).ceil() as usize;

    println!(
        "Training on {} tokens | Vocab: {} | Batches/epoch: {} | LR: {:.2e}",
        all_tokens.len(), tokenizer.vocab_size, total_steps_per_epoch, learning_rate
    );

    for epoch in 0..epochs {
        let mut total_loss = 0f32;
        let mut steps_ok   = 0usize;
        let mut steps_nan  = 0usize;

        for (batch_idx, batch_chunks) in chunks.chunks(batch_size).enumerate() {
            let mut input_data:  Vec<i32> = Vec::new();
            let mut target_data: Vec<i32> = Vec::new();

            for chunk in batch_chunks {
                if chunk.len() < seq_len + 1 { continue; }
                for &t in &chunk[..seq_len]    { input_data.push(t as i32); }
                for &t in &chunk[1..=seq_len]  { target_data.push(t as i32); }
            }
            if input_data.is_empty() { continue; }

            let real_batch = input_data.len() / seq_len;
            let input = Tensor::<TrainBackend, 2, Int>::from_data(
                TensorData::new(input_data, [real_batch, seq_len]), &device,
            );
            let targets = Tensor::<TrainBackend, 2, Int>::from_data(
                TensorData::new(target_data, [real_batch, seq_len]), &device,
            );

            let (_logits, loss_opt) = model.forward(input, Some(targets));
            let loss = loss_opt.expect("loss should be Some when targets provided");
            let loss_val: f32 = loss.clone().into_scalar();

            if batch_idx % 10 == 0 || batch_idx + 1 == total_steps_per_epoch {
                let pct = (batch_idx as f32 / total_steps_per_epoch as f32) * 100.0;
                print!(
                    "\rEpoch [{}/{}] Step [{}/{}] ({:.1}%) Loss: {:.4}  NaN steps: {}",
                    epoch + 1, epochs, batch_idx + 1, total_steps_per_epoch,
                    pct, loss_val, steps_nan
                );
                let _ = std::io::stdout().flush();
            }

            if loss_val.is_nan() || loss_val.is_infinite() {
                steps_nan += 1;
                continue;
            }

            total_loss += loss_val;
            steps_ok   += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads);
        }

        println!();
        if steps_ok > 0 {
            let avg = total_loss / steps_ok as f32;
            println!(
                "Epoch {} done. Avg loss: {:.4}  ({} ok, {} NaN)",
                epoch + 1, avg, steps_ok, steps_nan
            );
        } else {
            println!(
                "Epoch {} — ALL {} steps returned NaN. \
                 Check model init / learning rate.",
                epoch + 1, steps_nan
            );
        }

        if let Err(e) = save_model(&model, SAVE_PATH) {
            eprintln!("Warning: checkpoint save failed: {e}");
        }
    }

    model
}
