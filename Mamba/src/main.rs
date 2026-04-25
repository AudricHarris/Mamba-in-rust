// The wgpu backend has deeply nested types that exceed Rust's default
// recursion limit of 128. This is the official burn book fix.
#![recursion_limit = "256"]

mod data;
mod model;
mod tokenizer;
mod training;

use crate::data::DataBatcher;
use crate::model::{ModelArgs, MambaNlp};
use crate::tokenizer::Tokenizer;
use crate::training::{train, load_model, SAVE_PATH};
use burn::module::AutodiffModule;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dataset::Dataset;
use std::io::{self, BufRead, Write};
use std::path::Path;

fn prompt_usize(stdin: &mut impl BufRead, msg: &str) -> Option<usize> {
    print!("{msg}");
    io::stdout().flush().ok();
    let mut line = String::new();
    stdin.read_line(&mut line).ok();
    line.trim().parse().ok()
}

fn prompt_f64(stdin: &mut impl BufRead, msg: &str) -> Option<f64> {
    print!("{msg}");
    io::stdout().flush().ok();
    let mut line = String::new();
    stdin.read_line(&mut line).ok();
    line.trim().parse().ok()
}

fn prompt_yn(stdin: &mut impl BufRead, msg: &str) -> bool {
    print!("{msg} [y/N]: ");
    io::stdout().flush().ok();
    let mut line = String::new();
    stdin.read_line(&mut line).ok();
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
}

fn main() -> anyhow::Result<()> {
    // ── Thread pool ──────────────────────────────────────────────────────────
    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get()).unwrap_or(4);
    let use_cores = (total_cores / 2).max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(use_cores).build_global().ok();
    println!("Using {use_cores}/{total_cores} CPU threads.");

    // ── GPU device confirmation ──────────────────────────────────────────────
    // WgpuDevice::default() selects the best available GPU via wgpu.
    // On systems with a discrete GPU it picks that; otherwise falls back to
    // the integrated GPU or CPU-side Vulkan/Metal/DX12 adapter.
    // The burn wgpu backend compiles all ops to GPU shader programs (WGSL),
    // so training and inference both run on your GPU automatically.
    let device = WgpuDevice::default();
    println!("Backend: burn-wgpu (GPU). Device: {:?}", device);

    let db_path = "data/buffer.sqlite";

    // ── Dataset ──────────────────────────────────────────────────────────────
    DataBatcher::stream_to_sqlite(
        "https://huggingface.co/datasets/ameykaran/english-text-corpus/resolve/main/eng-000.jsonl",
        db_path,
        Some(1000),
    )?;
    let dataset  = DataBatcher::load_buffered_dataset(db_path)?;
    let texts: Vec<String> = (0..dataset.len())
        .filter_map(|i| dataset.get(i))
        .map(|item| item.text)
        .collect();
    let tokenizer = Tokenizer::build_from_texts(&texts);

    let total_tokens: usize = texts.iter()
        .map(|t| tokenizer.encode(t).len() + 1)
        .sum();
    println!("Dataset ready: {} samples, {} tokens available.", texts.len(), total_tokens);

    let args = ModelArgs { vocab_size: tokenizer.vocab_size, ..ModelArgs::default() };

    // ── Training control ─────────────────────────────────────────────────────
    let (skip_training, max_tokens, epochs, learning_rate) = {
        let stdin_raw = io::stdin();
        let mut stdin = stdin_raw.lock();

        let checkpoint_exists = Path::new(&format!("{SAVE_PATH}.mpk.gz")).exists();

        let skip = if checkpoint_exists {
            println!("Existing checkpoint found at {SAVE_PATH}.");
            prompt_yn(&mut stdin, "Skip training and load checkpoint?")
        } else {
            false
        };

        let (mt, ep, lr) = if skip {
            (None, 0, 0.0)
        } else {
            let mt = {
                print!("How many tokens to train on? (Enter = all {total_tokens}): ");
                io::stdout().flush()?;
                let mut line = String::new();
                stdin.read_line(&mut line)?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    println!("Using all {total_tokens} tokens.");
                    None
                } else {
                    match trimmed.parse::<usize>() {
                        Ok(n) if n >= 1 && n <= total_tokens => { println!("Using {n} tokens."); Some(n) }
                        Ok(n) if n > total_tokens => { println!("Capped to {total_tokens}."); None }
                        _ => { println!("Invalid — using all tokens."); None }
                    }
                }
            };

            let ep = prompt_usize(&mut stdin, "Epochs? (Enter = 3): ").unwrap_or(3);
            println!("Training for {ep} epochs.");

            let lr = prompt_f64(&mut stdin, "Learning rate? (Enter = 0.0001): ").unwrap_or(1e-4);
            println!("Learning rate: {lr:.2e}");

            (mt, ep, lr)
        };

        (skip, mt, ep, lr)
    };

    // ── Train or load ────────────────────────────────────────────────────────
    let inf_model: MambaNlp<Wgpu> = if skip_training {
        let train_model = load_model(&args, training::SAVE_PATH, &device)?;
        train_model.valid()
    } else {
        println!("Starting training…");
        let train_model = train(
            &texts, &tokenizer, &args,
            epochs, 128, 2, max_tokens, learning_rate,
        );
        train_model.valid()
    };

    // ── Inference REPL ───────────────────────────────────────────────────────
    // Generation parameters — tweak these to taste:
    //   temperature : 0.7 = focused, 1.0 = balanced, 1.3 = creative
    //   top_k       : 40–100 keeps vocabulary diverse but coherent
    //   rep_penalty : 0.85 strongly discourages repeats; 0.95 is gentler
    let gen_temp: f64    = 1.0;
    let gen_top_k: usize = 50;
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
        if trimmed.is_empty() { break; }

        let encoded: Vec<i32> = tokenizer.encode(trimmed)
            .into_iter().map(|x| x as i32).collect();
        let seq_len = encoded.len();

        let input_ids = Tensor::<Wgpu, 2, Int>::from_data(
            TensorData::new(encoded, [1, seq_len]), &device,
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
            tokenizer.decode(&tokens.iter().map(|&x| x as usize).collect::<Vec<_>>())
        );
    }

    println!("Bye!");
    Ok(())
}
