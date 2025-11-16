mod data;

use crate::data::{DataBatcher, TextItem};
use burn_dataset::Dataset;

fn main() -> anyhow::Result<()> {
   let url =
        "https://huggingface.co/datasets/ameykaran/english-text-corpus/resolve/main/eng-000.jsonl";
    let db_path = "data/buffer.sqlite";

    let max_samples = Some(5000);
    
    println!("Streaming dataset into SQLite…");
    // DataBatcher::stream_to_sqlite(url, db_path, max_samples)?;
    
    println!("Loading buffered dataset…");
    let dataset = DataBatcher::load_buffered_dataset(db_path)?;
    
    println!("Printing example sample:");
    DataBatcher::print_samples(&dataset, 1000);
    let count = dataset.len();
    
    println!("Buffered dataset contains {} samples.", count);

    Ok(())
}
