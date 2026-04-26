#![recursion_limit = "256"]

mod data;
mod model;
mod tokenizer;
mod training;

use crate::data::DataBatcher;
use crate::model::{ModelArgs, MambaNlp};
use crate::tokenizer::Tokenizer;
use crate::training::{train, train_from_checkpoint, load_model, SAVE_PATH};
use burn::module::AutodiffModule;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dataset::Dataset;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;


fn prompt_line(stdin: &mut impl BufRead, msg: &str) -> String {
    print!("{msg}");
    io::stdout().flush().ok();
    let mut line = String::new();
    stdin.read_line(&mut line).ok();
    line.trim().to_string()
}

fn prompt_usize(stdin: &mut impl BufRead, msg: &str) -> Option<usize> {
    prompt_line(stdin, msg).parse().ok()
}

fn prompt_f64(stdin: &mut impl BufRead, msg: &str) -> Option<f64> {
    prompt_line(stdin, msg).parse().ok()
}


fn list_checkpoints() -> Vec<String> {
    let dir = Path::new("checkpoints");
    if !dir.exists() {
        return vec![];
    }
    let mut found: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".mpk") {
                let stem = name.trim_end_matches(".mpk").to_string();
                Some(format!("checkpoints/{stem}"))
            } else {
                None
            }
        })
        .collect();
    found.sort();
    found
}

/// Prompt the user to pick a checkpoint from the list.
fn pick_checkpoint(stdin: &mut impl BufRead, checkpoints: &[String]) -> String {
    println!();
    println!("Available checkpoints:");
    for (i, cp) in checkpoints.iter().enumerate() {
        println!("  [{}] {}.mpk.gz", i + 1, cp);
    }
    let idx = loop {
        let raw = prompt_line(stdin, &format!("Select [1–{}]: ", checkpoints.len()));
        match raw.parse::<usize>() {
            Ok(n) if n >= 1 && n <= checkpoints.len() => break n - 1,
            _ => println!("Enter a number between 1 and {}.", checkpoints.len()),
        }
    };
    checkpoints[idx].clone()
}

/// Prompt for training hyper-params (tokens, epochs, lr).
fn prompt_training_params(
    stdin: &mut impl BufRead,
    total_tokens: usize,
    default_lr: f64,
) -> (Option<usize>, usize, f64) {
    let max_tokens = {
        let raw = prompt_line(
            stdin,
            &format!("Tokens to train on? (Enter = all {total_tokens}): "),
        );
        if raw.is_empty() {
            println!("Using all {total_tokens} tokens.");
            None
        } else {
            match raw.parse::<usize>() {
                Ok(n) if n >= 1 && n <= total_tokens => {
                    println!("Using {n} tokens.");
                    Some(n)
                }
                Ok(_) => { println!("Out of range — using all tokens."); None }
                Err(_) => { println!("Invalid — using all tokens."); None }
            }
        }
    };

    let epochs = prompt_usize(stdin, "Epochs? (Enter = 3): ").unwrap_or(3);
    println!("Training for {epochs} epochs.");

    let lr_prompt = format!("Learning rate? (Enter = {default_lr:.0e}): ");
    let learning_rate = prompt_f64(stdin, &lr_prompt).unwrap_or(default_lr);
    println!("Learning rate: {learning_rate:.2e}");

    (max_tokens, epochs, learning_rate)
}


enum StartChoice {
    Train { max_tokens: Option<usize>, epochs: usize, learning_rate: f64 },
    LoadCheckpoint { path: String },
    FineTune { path: String, max_tokens: Option<usize>, epochs: usize, learning_rate: f64 },
}

fn startup_menu(stdin: &mut impl BufRead, total_tokens: usize) -> StartChoice {
    let checkpoints = list_checkpoints();
    let has_ckpts = !checkpoints.is_empty();

    println!();
    println!("┌──────────────────────────────────────┐");
    println!("│           Mamba LM — Menu            │");
    println!("├──────────────────────────────────────┤");
    println!("│  1) Train from scratch / resume      │");
    if has_ckpts {
        println!("│  2) Train from existing checkpoint   │");
        println!("│  3) Load checkpoint (inference only) │");
    }
    println!("└──────────────────────────────────────┘");

    let max_choice = if has_ckpts { 3 } else { 1 };
    let choice = loop {
        let raw = prompt_line(stdin, "Choice: ");
        match raw.as_str() {
            "1" => break 1usize,
            "2" if has_ckpts => break 2,
            "3" if has_ckpts => break 3,
            _ => println!(
                "Please enter a number between 1 and {max_choice}."
            ),
        }
    };

    match choice {
        2 => {
            let path = pick_checkpoint(stdin, &checkpoints);
            println!();
            println!("Fine-tuning from: {path}.mpk.gz");
            println!("Tip: use a lower learning rate (e.g. 1e-5) to preserve learned weights.");
            let (max_tokens, epochs, learning_rate) =
                prompt_training_params(stdin, total_tokens, 1e-5);
            StartChoice::FineTune { path, max_tokens, epochs, learning_rate }
        }
        3 => {
            let path = pick_checkpoint(stdin, &checkpoints);
            StartChoice::LoadCheckpoint { path }
        }
        _ => {
            println!();
            let (max_tokens, epochs, learning_rate) =
                prompt_training_params(stdin, total_tokens, 1e-4);
            StartChoice::Train { max_tokens, epochs, learning_rate }
        }
    }
}


fn main() -> anyhow::Result<()> {
    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let use_cores = (total_cores / 2).max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(use_cores)
        .build_global()
        .ok();
    println!("Using {use_cores}/{total_cores} CPU threads.");

    let device = WgpuDevice::default();
    println!("Backend: burn-wgpu (GPU). Device: {:?}", device);

    let db_path = "data/buffer.sqlite";

    DataBatcher::stream_to_sqlite(
        "https://huggingface.co/datasets/ameykaran/english-text-corpus/resolve/main/eng-000.jsonl",
        db_path,
        Some(1000),
    )?;
    let dataset = DataBatcher::load_buffered_dataset(db_path)?;
    let texts: Vec<String> = (0..dataset.len())
        .filter_map(|i| dataset.get(i))
        .map(|item| item.text)
        .collect();
    let tokenizer = Tokenizer::build_from_texts(&texts);

    let total_tokens: usize = texts
        .iter()
        .map(|t| tokenizer.encode(t).len() + 1)
        .sum();
    println!(
        "Dataset ready: {} samples, {} tokens available.",
        texts.len(),
        total_tokens
    );

    let args = ModelArgs {
        vocab_size: tokenizer.vocab_size,
        ..ModelArgs::default()
    };

    let inf_model: MambaNlp<Wgpu> = {
        let stdin_raw = io::stdin();
        let mut stdin = stdin_raw.lock();
        match startup_menu(&mut stdin, total_tokens) {
            StartChoice::Train { max_tokens, epochs, learning_rate } => {
                println!("Starting training…");
                train(
                    &texts, &tokenizer, &args,
                    epochs, 128, 2, max_tokens, learning_rate,
                )
                .valid()
            }
            StartChoice::FineTune { path, max_tokens, epochs, learning_rate } => {
                println!("Starting fine-tuning…");
                train_from_checkpoint(
                    &path,
                    &texts, &tokenizer, &args,
                    epochs, 128, 2, max_tokens, learning_rate,
                )?
                .valid()
            }
            StartChoice::LoadCheckpoint { path } => {
                println!("Loading checkpoint: {path}…");
                load_model(&args, &path, &device)?.valid()
            }
        }
    };

    let gen_temp: f64        = 1.0;
    let gen_top_k: usize     = 50;
    let gen_rep_penalty: f32 = 0.85;

    println!("\nReady. Type a prompt (empty line to quit).");
    println!("(temp={gen_temp}, top_k={gen_top_k}, rep_penalty={gen_rep_penalty})");

    let stdin_raw = io::stdin();
    let mut stdin = stdin_raw.lock();
    loop {
        print!("You: ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        stdin.read_line(&mut prompt)?;
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            break;
        }

        let encoded: Vec<i32> = tokenizer
            .encode(trimmed)
            .into_iter()
            .map(|x| x as i32)
            .collect();
        let seq_len = encoded.len();

        let input_ids = Tensor::<Wgpu, 2, Int>::from_data(
            TensorData::new(encoded, [1, seq_len]),
            &device,
        );

        let output = inf_model.generate(
            input_ids,
            20,
            gen_temp,
            gen_top_k,
            gen_rep_penalty,
            &device,
        );
        let tokens: Vec<i32> = output.into_data().to_vec().unwrap();
        println!(
            "AI: {}\n",
            tokenizer.decode(
                &tokens.iter().map(|&x| x as usize).collect::<Vec<_>>()
            )
        );
    }

    println!("Bye!");
    Ok(())
}
