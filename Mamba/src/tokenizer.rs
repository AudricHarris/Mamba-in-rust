// ============================================================
// File: tokenizer.rs
// Developer: Audric HARRIS
// Update Date: 27/04/2026
// ============================================================

use tokenizers::{
    decoders::bpe::BPEDecoder,
    models::bpe::{BpeTrainerBuilder, BPE},
    normalizers::BertNormalizer,
    pre_tokenizers::whitespace::Whitespace,
    processors::template::{Template, TemplateProcessing},
    AddedToken, TokenizerBuilder,
    Tokenizer as HFTokenizer,
    TokenizerImpl,
};

#[derive(Clone, Debug)]
pub struct Tokenizer {
    inner: HFTokenizer,
    pub vocab_size: usize,
}

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const BOS_ID: u32 = 3;

impl Tokenizer {
    pub fn build_from_texts(texts: &[String], vocab_size: usize) -> anyhow::Result<Self> {
        let tmp_path = std::env::temp_dir().join("mamba_bpe_corpus.txt");
        std::fs::write(&tmp_path, texts.join("\n"))
            .map_err(|e| anyhow::anyhow!("Failed to write BPE training corpus: {e}"))?;

        let special_tokens = vec![
            AddedToken::from("<PAD>", true),
            AddedToken::from("<UNK>", true),
            AddedToken::from("<EOS>", true),
            AddedToken::from("<BOS>", true),
        ];

        let mut trainer = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(special_tokens)
            .build();

        let post_processor = TemplateProcessing::builder()
            .single(
                Template::try_from("<BOS>:3 $A:0 <EOS>:2")
                    .map_err(|e| anyhow::anyhow!("Template single failed: {e}"))?,
            )
            .pair(
                Template::try_from("<BOS>:3 $A:0 <EOS>:2 $B:1 <EOS>:2")
                    .map_err(|e| anyhow::anyhow!("Template pair failed: {e}"))?,
            )
            .special_tokens(vec![("<BOS>", BOS_ID), ("<EOS>", EOS_ID)])
            .build()
            .map_err(|e| anyhow::anyhow!("TemplateProcessing build failed: {e}"))?;

        let mut typed_tokenizer: TokenizerImpl<BPE, BertNormalizer, Whitespace, TemplateProcessing, BPEDecoder> =
            TokenizerBuilder::new()
                .with_model(BPE::default())
                .with_normalizer(Some(BertNormalizer::default()))
                .with_pre_tokenizer(Some(Whitespace))
                .with_post_processor(Some(post_processor))
                .build()
                .map_err(|e| anyhow::anyhow!("TokenizerBuilder failed: {e}"))?;

        typed_tokenizer
            .train_from_files(&mut trainer, vec![
                tmp_path.to_str().unwrap().to_string(),
            ])
            .map_err(|e| anyhow::anyhow!("BPE training failed: {e}"))?;

        let inner: HFTokenizer = typed_tokenizer.into();
        let actual_vocab = inner.get_vocab_size(true);
        Ok(Self { inner, vocab_size: actual_vocab })
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        self.inner.save(path, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer save failed: {e}"))?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let inner = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {e}"))?;
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self { inner, vocab_size })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.inner
            .encode(text, true)
            .map(|enc| enc.get_ids().iter().map(|&id| id as usize).collect())
            .unwrap_or_default()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&u32_ids, true).unwrap_or_default()
    }
}
