// ============================================================
// File: data.rs
// Developer: Audric HARRIS
// Create Date: 6/11/2025
// Update Date: 13/11/2025
// Objective: Stream Hugging Face text dataset into SQLite buffer
// ============================================================

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use burn::tensor::{Tensor, backend::Backend};
use burn_dataset::{Dataset, SqliteDataset};
use reqwest::blocking::Client;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::*;

// ------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct TextItem {
    pub text: String,
}

#[derive(Clone, Default)]
pub struct DataBatcher {}

#[derive(Clone)]
pub struct DataBatch<B: Backend> {
    pub text: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

// ------------------------------------------------------------
// Streaming and Buffering Logic
// ------------------------------------------------------------

impl DataBatcher {
    pub fn stream_to_sqlite(
        url: &str,
        db_path: &str,
        max_samples: Option<usize>,
    ) -> anyhow::Result<()> {
        println!("Starting stream from: {}", url);

        if let Some(parent) = Path::new(db_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS dataset (row_id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)",
            [],
        )?;

        let client = Client::new();
        let response = client.get(url).send()?;
        let reader = BufReader::new(response);

        let mut inserted = 0;
        for line in reader.lines().flatten() {
            if let Ok(item) = serde_json::from_str::<TextItem>(&line) {
                conn.execute("INSERT INTO dataset (text) VALUES (?1)", params![item.text])?;
                inserted += 1;

                if let Some(limit) = max_samples {
                    if inserted >= limit {
                        println!("Reached limit of {} samples, stopping stream.", limit);
                        break;
                    }
                }

                if inserted % 1000 == 0 {
                    println!("Inserted {} samples...", inserted);
                }
            }
        }

        println!("Stream completed. Total samples buffered: {}", inserted);
        Ok(())
    }

    /// Load the buffered dataset from SQLite.
    pub fn load_buffered_dataset(db_path: &str) -> anyhow::Result<SqliteDataset<TextItem>> {
        let dataset = SqliteDataset::<TextItem>::from_db_file(db_path, "dataset")?;
        Ok(dataset)
    }

    /// Convenience: print a random sample from the dataset.
    pub fn print_samples(dataset: &SqliteDataset<TextItem>, count: usize) {
        println!("Printing first {count} samples:");

        for (i, sample) in dataset.iter().take(count).enumerate() {
            println!("{:4}: {:?}", i + 1, sample);
        }
    }
}
