# 🐍 Mamba in Rust

**Mamba in Rust** provides a from-scratch implementation of the **Mamba State Space Model (SSM)** architecture, powered by the [**Burn**](https://github.com/burn-rs/burn) deep learning framework.  

Mamba is a *selective SSM* designed as an efficient alternative to Transformers achieving **linear-time scaling** with sequence length while maintaining competitive performance on language modeling, time-series forecasting, and beyond.  

By leveraging **Rust’s performance and safety**, this project aims for **fast inference**, **efficient training**, and **seamless deployment**, including **WebAssembly (WASM)** support for browser-based machine learning.

---

## 🧠 Why This Project?

- ⚡ **Efficiency:** Mamba avoids the quadratic complexity of Transformers, making it ideal for long-context or streaming tasks.  
- 🦀 **Rust Advantages:** Zero-cost abstractions enable high-speed execution without garbage collection. Rust’s memory safety eliminates common ML bugs.  
- 🔥 **Burn Integration:** Unified CPU/GPU acceleration, modular layers, and ecosystem compatibility for rapid experimentation.

> **Note:** This is a **work-in-progress prototype**, rebuilt from an earlier PyTorch version (Aug–Sep 2024). Expect continuous improvements based on new research and hands-on testing. Pre-trained weights and demo notebooks will be published on **Kaggle** soon.

---

## ✨ Features

- **Core Modules:** Selective SSM layers, hardware-aware recurrent kernels, and sequence modeling primitives.  
- **Training Pipeline:** Scripts for benchmarking on synthetic data, toy language tasks, and real-world datasets.  
- **Portability:** Compile to **WebAssembly** for edge devices or browser demos.  
- **Extensibility:** Easily switch backends (e.g., CUDA, Metal) through Burn’s backend abstraction.

---

## 🚀 Quick Start

### Prerequisites
- **Rust** `1.75+` (install via [rustup](https://rustup.rs))  
- **Burn** deep learning framework (added as a Cargo dependency)

### Installation

```bash
git clone https://github.com/AudricHarris/mamba-in-rust.git
cd mamba-in-rust
cargo build --release
