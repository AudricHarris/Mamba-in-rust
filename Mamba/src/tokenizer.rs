use std::collections::HashMap;

pub struct Tokenizer {
    pub vocab: HashMap<String, usize>,
    pub reverse_vocab: Vec<String>,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn build_from_texts(texts: &[String]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();

        vocab.insert("<PAD>".to_string(), 0);
        vocab.insert("<UNK>".to_string(), 1);
        vocab.insert("<EOS>".to_string(), 2);

        let mut idx = 3usize;
        for text in texts {
            for word in text.split_whitespace() {
                let w = word.to_lowercase();
                if !vocab.contains_key(&w) {
                    vocab.insert(w, idx);
                    idx += 1;
                }
            }
        }

        let mut reverse_vocab = vec![String::new(); vocab.len()];
        for (word, i) in &vocab {
            reverse_vocab[*i] = word.clone();
        }

        let vocab_size = vocab.len();
        Self { vocab, reverse_vocab, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| {
                let w = w.to_lowercase();
                *self.vocab.get(&w).unwrap_or(&1)
            })
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != 0 && id != 2)
            .map(|&id| {
                self.reverse_vocab
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| "<UNK>".to_string())
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}
