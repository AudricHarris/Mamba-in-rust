mod data;

use crate::data::{DataBatcher, TextItem};
use burn_dataset::Dataset;

fn main() -> anyhow::Result<()> {
    // ------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------
    let url =
        "https://huggingface.co/datasets/ameykaran/english-text-corpus/resolve/main/eng-000.jsonl";
    let db_path = "data/buffer.sqlite";

    // How many samples to download (None = full stream)
    let max_samples = Some(5000);

    // ------------------------------------------------------------
    // Stream and store dataset
    // ------------------------------------------------------------
    println!("Streaming dataset into SQLite…");
    DataBatcher::stream_to_sqlite(url, db_path, max_samples)?;

    // ------------------------------------------------------------
    // Load dataset
    // ------------------------------------------------------------
    println!("Loading buffered dataset…");
    let dataset = DataBatcher::load_buffered_dataset(db_path)?;

    // ------------------------------------------------------------
    // Print a sample
    // ------------------------------------------------------------
    println!("Printing example sample:");
    DataBatcher::print_samples(&dataset, 1000);

    // ------------------------------------------------------------
    // Confirm dataset length
    // ------------------------------------------------------------
    let count = dataset.len();
    println!("Buffered dataset contains {} samples.", count);

    Ok(())
}
