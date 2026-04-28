#![recursion_limit = "256"]

mod data;
mod model;
mod tokenizer;
mod training;

use crate::data::{TextItem, split_dataset};
use crate::model::{ModelArgs, MambaNlp};
use crate::tokenizer::Tokenizer;
use crate::training::{train, train_from_checkpoint, load_model};
use burn::module::AutodiffModule;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dataset::{Dataset, InMemDataset, SqliteDataset};
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

fn prompt_training_params(
    stdin: &mut impl BufRead,
    total_tokens: usize,
    default_lr: f64,
) -> (usize, f64) {
    let epochs = prompt_usize(stdin, "Epochs? (Enter = 3): ").unwrap_or(3);
    println!("Training for {epochs} epochs.");
    let lr_prompt = format!("Learning rate? (Enter = {default_lr:.0e}): ");
    let learning_rate = prompt_f64(stdin, &lr_prompt).unwrap_or(default_lr);
    println!("Learning rate: {learning_rate:.2e}");
    let _ = total_tokens;
    (epochs, learning_rate)
}

enum StartChoice {
    Train { epochs: usize, learning_rate: f64 },
    LoadCheckpoint { path: String },
    FineTune { path: String, epochs: usize, learning_rate: f64 },
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
            _ => println!("Please enter a number between 1 and {max_choice}."),
        }
    };

    match choice {
        2 => {
            let path = pick_checkpoint(stdin, &checkpoints);
            println!();
            println!("Fine-tuning from: {path}.mpk.gz");
            let (epochs, learning_rate) = prompt_training_params(stdin, total_tokens, 1e-5);
            StartChoice::FineTune { path, epochs, learning_rate }
        }
        3 => {
            let path = pick_checkpoint(stdin, &checkpoints);
            StartChoice::LoadCheckpoint { path }
        }
        _ => {
            println!();
            let (epochs, learning_rate) = prompt_training_params(stdin, total_tokens, 1e-4);
            StartChoice::Train { epochs, learning_rate }
        }
    }
}

// ------------ //
// Data helpers //
// ------------ //

fn stream_to_sqlite(url: &str, db_path: &str, limit: Option<usize>) -> anyhow::Result<()> {
    use std::io::BufReader;

    if Path::new(db_path).exists() {
        println!("SQLite buffer already exists at {db_path}, skipping download.");
        return Ok(());
    }
    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    println!("Streaming dataset from {url} …");
    let response = ureq::get(url).call()?;
    let reader = BufReader::new(response.into_reader());

    let conn = rusqlite::Connection::open(db_path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS data (
            row_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        );"
    )?;

    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(&line)?;
        if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
            conn.execute("INSERT INTO data (text) VALUES (?1)", [text])?;
            count += 1;
        }
        if let Some(lim) = limit {
            if count >= lim { break; }
        }
    }
    println!("Inserted {count} rows into {db_path}.");
    Ok(())
}

fn load_buffered_dataset(db_path: &str) -> anyhow::Result<SqliteDataset<TextItem>> {
    SqliteDataset::<TextItem>::from_db_file(db_path, "data")
        .map_err(|e| anyhow::anyhow!("Failed to open dataset: {e}"))
}

fn split_to_mem(
    db_path: &str,
    valid_frac: f64,
) -> anyhow::Result<(InMemDataset<TextItem>, InMemDataset<TextItem>)> {
    let ds = load_buffered_dataset(db_path)?;
    let (train_items, valid_items) = split_dataset(&ds, valid_frac);
    Ok((InMemDataset::new(train_items), InMemDataset::new(valid_items)))
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

    stream_to_sqlite(
        "https://huggingface.co/datasets/ameykaran/english-text-corpus/resolve/main/eng-000.jsonl",
        db_path,
        Some(1000),
    )?;

    let dataset = load_buffered_dataset(db_path)?;
    let texts: Vec<String> = (0..dataset.len())
        .filter_map(|i| dataset.get(i))
        .map(|item: TextItem| item.text)
        .collect();

    let tokenizer = Tokenizer::build_from_texts(&texts, 8_000)?;

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
            StartChoice::Train { epochs, learning_rate } => {
                println!("Starting training…");
                let (train_ds, valid_ds) = split_to_mem(db_path, 0.1)?;
                train(
                    train_ds,
                    valid_ds,
                    tokenizer.clone(),
                    &args,
                    epochs, 128, 8,
                    learning_rate,
                )?
                .valid()
            }
            StartChoice::FineTune { path, epochs, learning_rate } => {
                println!("Starting fine-tuning…");
                let (train_ds, valid_ds) = split_to_mem(db_path, 0.1)?;
                train_from_checkpoint(
                    &path,
                    train_ds,
                    valid_ds,
                    tokenizer.clone(),
                    &args,
                    epochs, 128, 8,
                    learning_rate,
                )?
                .valid()
            }
            StartChoice::LoadCheckpoint { path } => {
                println!("Loading checkpoint: {path}…");
                load_model(&args, &path, &device)?.valid()
            }
        }
    };

    let gen_temp: f64        = 0.9;
    let gen_top_p: f64       = 0.9;
    let gen_rep_penalty: f32 = 1.2;

    println!("\nReady. Type a prompt (empty line to quit).");
    println!("(temp={gen_temp}, top_p={gen_top_p}, rep_penalty={gen_rep_penalty})");

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
            gen_top_p,
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
