#![recursion_limit = "256"]

mod data;
mod model;
mod tokenizer;
mod training;

use crate::data::{AlpacaItem, TextItem, split_dataset, load_jsonl_items};
use crate::model::{ModelArgs, MambaNlp};
use crate::tokenizer::Tokenizer;
use crate::training::{train, train_from_checkpoint, load_model};
use burn::module::AutodiffModule;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dataset::InMemDataset;
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
    default_lr: f64,
) -> (usize, f64) {
    let epochs = prompt_usize(stdin, "Epochs? (Enter = 3): ").unwrap_or(3);
    println!("Training for {epochs} epochs.");
    let lr_prompt = format!("Learning rate? (Enter = {default_lr:.0e}): ");
    let learning_rate = prompt_f64(stdin, &lr_prompt).unwrap_or(default_lr);
    println!("Learning rate: {learning_rate:.2e}");
    (epochs, learning_rate)
}


enum StartChoice {
    Pretrain {
        pretrain_epochs: usize,
        pretrain_lr: f64,
        finetune: bool,
        finetune_epochs: usize,
        finetune_lr: f64,
    },
    TrainAlpaca { epochs: usize, learning_rate: f64 },
    FineTune { path: String, epochs: usize, learning_rate: f64 },
    LoadCheckpoint { path: String },
}

fn startup_menu(stdin: &mut impl BufRead) -> StartChoice {
    let checkpoints = list_checkpoints();
    let has_ckpts = !checkpoints.is_empty();

    println!();
    println!("┌──────────────────────────────────────────────────┐");
    println!("│              Mamba LM — Menu                     │");
    println!("├──────────────────────────────────────────────────┤");
    println!("│  1) Pretrain on English corpus (+optional Alpaca)│");
    println!("│  2) Fine-tune on Alpaca (scratch / auto-resume)  │");
    if has_ckpts {
        println!("│  3) Fine-tune from existing checkpoint           │");
        println!("│  4) Load checkpoint (inference only)             │");
    }
    println!("└──────────────────────────────────────────────────┘");

    let max_choice = if has_ckpts { 4 } else { 2 };
    let choice = loop {
        let raw = prompt_line(stdin, "Choice: ");
        match raw.as_str() {
            "1" => break 1usize,
            "2" => break 2,
            "3" if has_ckpts => break 3,
            "4" if has_ckpts => break 4,
            _ => println!("Please enter a number between 1 and {max_choice}."),
        }
    };

    match choice {
        1 => {
            println!("\n── Pretraining on English corpus ──");
            let (pretrain_epochs, pretrain_lr) = prompt_training_params(stdin, 1e-4);
            let do_ft = matches!(
                prompt_line(stdin, "Also fine-tune on Alpaca afterwards? [y/N]: ")
                    .to_lowercase()
                    .as_str(),
                "y" | "yes"
            );
            let (finetune_epochs, finetune_lr) = if do_ft {
                println!("\n── Alpaca fine-tuning params ──");
                prompt_training_params(stdin, 1e-5)
            } else {
                (0, 0.0)
            };
            StartChoice::Pretrain {
                pretrain_epochs,
                pretrain_lr,
                finetune: do_ft,
                finetune_epochs,
                finetune_lr,
            }
        }
        3 if has_ckpts => {
            let path = pick_checkpoint(stdin, &checkpoints);
            println!("\nFine-tuning from: {path}.mpk.gz");
            let (epochs, learning_rate) = prompt_training_params(stdin, 1e-5);
            StartChoice::FineTune { path, epochs, learning_rate }
        }
        4 if has_ckpts => {
            let path = pick_checkpoint(stdin, &checkpoints);
            StartChoice::LoadCheckpoint { path }
        }
        _ => {
            println!();
            let (epochs, learning_rate) = prompt_training_params(stdin, 1e-4);
            StartChoice::TrainAlpaca { epochs, learning_rate }
        }
    }
}


const CORPUS_URL: &str =
    "https://huggingface.co/datasets/ameykaran/english-text-corpus/resolve/main/eng-000.jsonl";

const CORPUS_CACHE: &str = "data/eng-000.jsonl";

fn ensure_corpus_jsonl() -> anyhow::Result<()> {
    if Path::new(CORPUS_CACHE).exists() {
        println!("Corpus cache found at {CORPUS_CACHE}, skipping download.");
        return Ok(());
    }
    if let Some(parent) = Path::new(CORPUS_CACHE).parent() {
        std::fs::create_dir_all(parent)?;
    }
    println!("Downloading English text corpus …");
    let response = ureq::get(CORPUS_URL).call()?;
    let mut reader = response.into_reader();
    let tmp = format!("{CORPUS_CACHE}.tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        std::io::copy(&mut reader, &mut file)?;
    }
    std::fs::rename(&tmp, CORPUS_CACHE)?;
    println!("Saved to {CORPUS_CACHE}.");
    Ok(())
}

fn load_corpus_items(limit: Option<usize>) -> anyhow::Result<Vec<TextItem>> {
    ensure_corpus_jsonl()?;
    let items = load_jsonl_items(CORPUS_CACHE, limit)?;
    println!("Loaded {} corpus records.", items.len());
    Ok(items)
}


const ALPACA_URL: &str =
    "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json";

const ALPACA_CACHE: &str = "data/alpaca_data_cleaned.json";

fn ensure_alpaca_json() -> anyhow::Result<()> {
    if Path::new(ALPACA_CACHE).exists() {
        println!("Alpaca cache found at {ALPACA_CACHE}, skipping download.");
        return Ok(());
    }
    if let Some(parent) = Path::new(ALPACA_CACHE).parent() {
        std::fs::create_dir_all(parent)?;
    }
    println!("Downloading Alpaca-cleaned dataset …");
    let response = ureq::get(ALPACA_URL).call()?;
    let mut reader = response.into_reader();
    let tmp = format!("{ALPACA_CACHE}.tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        std::io::copy(&mut reader, &mut file)?;
    }
    std::fs::rename(&tmp, ALPACA_CACHE)?;
    println!("Saved to {ALPACA_CACHE}.");
    Ok(())
}

fn load_alpaca_items(limit: Option<usize>) -> anyhow::Result<Vec<TextItem>> {
    ensure_alpaca_json()?;
    let raw = std::fs::read_to_string(ALPACA_CACHE)?;
    let mut records: Vec<AlpacaItem> = serde_json::from_str(&raw)?;
    if let Some(lim) = limit {
        records.truncate(lim);
    }
    let items: Vec<TextItem> = records.into_iter().map(TextItem::from).collect();
    println!("Loaded {} Alpaca records.", items.len());
    Ok(items)
}

const TOKENIZER_PATH: &str = "checkpoints/tokenizer.json";

fn build_or_load_tokenizer(
    texts_for_training: &[String],
    vocab_size: usize,
) -> anyhow::Result<Tokenizer> {
    if Path::new(TOKENIZER_PATH).exists() {
        println!("Loading existing tokenizer from {TOKENIZER_PATH} …");
        let tok = Tokenizer::load(TOKENIZER_PATH)?;
        println!("Tokenizer loaded (vocab={})", tok.vocab_size);
        return Ok(tok);
    }
    println!("Building BPE tokenizer from {} texts …", texts_for_training.len());
    let tok = Tokenizer::build_from_texts(texts_for_training, vocab_size)?;
    if let Some(parent) = Path::new(TOKENIZER_PATH).parent() {
        std::fs::create_dir_all(parent)?;
    }
    tok.save(TOKENIZER_PATH)?;
    println!("Tokenizer saved to {TOKENIZER_PATH} (vocab={})", tok.vocab_size);
    Ok(tok)
}

const PRETRAIN_CHECKPOINT: &str = "checkpoints/mamba_pretrained";

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

    let inf_model: MambaNlp<Wgpu> = {
        let stdin_raw = io::stdin();
        let mut stdin = stdin_raw.lock();

        match startup_menu(&mut stdin) {

            StartChoice::Pretrain {
                pretrain_epochs,
                pretrain_lr,
                finetune,
                finetune_epochs,
                finetune_lr,
            } => {
                let corpus_items = load_corpus_items(Some(10_000))?;

                let alpaca_items_for_tok = if finetune {
                    println!("Pre-loading Alpaca texts for tokenizer vocabulary …");
                    load_alpaca_items(None)?
                } else {
                    vec![]
                };

                let all_texts_for_tok: Vec<String> = corpus_items
                    .iter()
                    .chain(alpaca_items_for_tok.iter())
                    .map(|t| t.text.clone())
                    .collect();
                let tokenizer = build_or_load_tokenizer(&all_texts_for_tok, 16_000)?;

                let args = ModelArgs {
                    vocab_size: tokenizer.vocab_size,
                    ..ModelArgs::default()
                };

                println!("\n═══════════════════════════════════════");
                println!(" PHASE 1 — Pretraining on English corpus");
                println!("═══════════════════════════════════════");
                let (corpus_train, corpus_valid) = split_dataset(corpus_items, 0.05);
                println!(
                    "Corpus split: {} train / {} valid",
                    corpus_train.len(), corpus_valid.len()
                );
                let pretrained = train(
                    InMemDataset::new(corpus_train),
                    InMemDataset::new(corpus_valid),
                    tokenizer.clone(),
                    &args,
                    pretrain_epochs,
                    128,
                    16,
                    pretrain_lr,
                )?;

                crate::training::save_model(&pretrained, PRETRAIN_CHECKPOINT)?;
                println!("Pretrain checkpoint saved to {PRETRAIN_CHECKPOINT}");

                if !finetune {
                    pretrained.valid()
                } else {
                    println!("\n═══════════════════════════════════════");
                    println!(" PHASE 2 — Fine-tuning on Alpaca");
                    println!("═══════════════════════════════════════");
                    let alpaca_items = if alpaca_items_for_tok.is_empty() {
                        load_alpaca_items(None)?
                    } else {
                        alpaca_items_for_tok
                    };
                    let (alpaca_train, alpaca_valid) = split_dataset(alpaca_items, 0.1);
                    println!(
                        "Alpaca split: {} train / {} valid",
                        alpaca_train.len(), alpaca_valid.len()
                    );
                    train_from_checkpoint(
                        PRETRAIN_CHECKPOINT,
                        InMemDataset::new(alpaca_train),
                        InMemDataset::new(alpaca_valid),
                        tokenizer.clone(),
                        &args,
                        finetune_epochs,
                        128,
                        16,
                        finetune_lr,
                    )?.valid()
                }
            }

            StartChoice::TrainAlpaca { epochs, learning_rate } => {
                let alpaca_items = load_alpaca_items(None)?;
                let all_texts: Vec<String> =
                    alpaca_items.iter().map(|t| t.text.clone()).collect();
                let tokenizer = build_or_load_tokenizer(&all_texts, 16_000)?;
                let args = ModelArgs {
                    vocab_size: tokenizer.vocab_size,
                    ..ModelArgs::default()
                };
                println!("Starting Alpaca training…");
                let (train_items, valid_items) = split_dataset(alpaca_items, 0.1);
                train(
                    InMemDataset::new(train_items),
                    InMemDataset::new(valid_items),
                    tokenizer.clone(),
                    &args,
                    epochs,
                    128,
                    16,
                    learning_rate,
                )?.valid()
            }

            StartChoice::FineTune { path, epochs, learning_rate } => {
                let alpaca_items = load_alpaca_items(None)?;
                let all_texts: Vec<String> =
                    alpaca_items.iter().map(|t| t.text.clone()).collect();
                let tokenizer = build_or_load_tokenizer(&all_texts, 16_000)?;
                let args = ModelArgs {
                    vocab_size: tokenizer.vocab_size,
                    ..ModelArgs::default()
                };
                println!("Starting fine-tuning from {path}…");
                let (train_items, valid_items) = split_dataset(alpaca_items, 0.1);
                train_from_checkpoint(
                    &path,
                    InMemDataset::new(train_items),
                    InMemDataset::new(valid_items),
                    tokenizer.clone(),
                    &args,
                    epochs,
                    256,
                    8,
                    learning_rate,
                )?.valid()
            }

            StartChoice::LoadCheckpoint { path } => {
                let tokenizer = Tokenizer::load(TOKENIZER_PATH).map_err(|e| {
                    anyhow::anyhow!(
                        "Cannot load tokenizer from {TOKENIZER_PATH}: {e}\n\
                         Run training at least once to build it."
                    )
                })?;
                let args = ModelArgs {
                    vocab_size: tokenizer.vocab_size,
                    ..ModelArgs::default()
                };
                println!("Loading checkpoint: {path}…");
                load_model(&args, &path, &device)?.valid()
            }
        }
    };

    let tokenizer = Tokenizer::load(TOKENIZER_PATH).unwrap_or_else(|_| {
        panic!("Tokenizer not found at {TOKENIZER_PATH}. Train the model first.");
    });

    let gen_temp: f64 = 0.9;
    let gen_top_p: f64 = 0.9;
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

        let formatted = format!("### Instruction:\n{trimmed}\n\n### Response:\n");

        let encoded: Vec<i32> = tokenizer
            .encode(&formatted)
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
            200,
            gen_temp,
            gen_top_p,
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
