# English

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
```

# French

# 🐍 Mamba en Rust

Mamba en Rust propose une implémentation from-scratch de l'architecture Mamba State Space Model (SSM), alimentée par le framework d'apprentissage profond [**Burn**](https://github.com/burn-rs/burn).

Mamba est un SSM sélectif conçu comme une alternative efficace aux Transformers, atteignant un échelle linéaire en fonction de la longueur de la séquence tout en maintenant des performances compétitives en modélisation de langage, prévision de séries temporelles, et plus encore.

En tirant parti de la performance et de la sécurité de Rust, ce projet vise une inférence rapide, un entraînement efficace, et un déploiement fluide, incluant un support pour WebAssembly (WASM) pour l'apprentissage automatique basé sur navigateur.

## 🧠 Pourquoi ce projet ?
- ⚡ Efficacité : Mamba évite la complexité quadratique des Transformers, ce qui le rend idéal pour les tâches à contexte long ou en streaming.
- 🦀 Avantages de Rust : Les abstractions sans coût zéro permettent une exécution à haute vitesse sans collecte de garbage. La sécurité mémoire de Rust élimine les bugs courants en ML.
- 🔥 Intégration Burn : Accélération unifiée CPU/GPU, couches modulaires, et compatibilité avec l'écosystème pour des expérimentations rapides.


> Note : Il s'agit d'un prototype en cours de développement, reconstruit à partir d'une version PyTorch antérieure (août–sept. 2024). Attendez-vous à des améliorations continues basées sur de nouvelles recherches et des tests pratiques. Des poids pré-entraînés et des notebooks de démonstration seront publiés sur Kaggle sous peu.

## ✨ Fonctionnalités

- Modules principaux : Couches SSM sélectives, noyaux récurrents adaptés au matériel, et primitives de modélisation de séquences.
- Pipeline d'entraînement : Scripts pour le benchmarking sur des données synthétiques, des tâches de langage simplifiées, et des ensembles de données réels.
- Portabilité : Compilation vers WebAssembly pour les appareils edge ou les démos en navigateur.
- Extensibilité : Basculer facilement vers d'autres backends (ex. : CUDA, Metal) grâce à l'abstraction backend de Burn.

## 🚀 Démarrage rapide

### Prérequis
- Rust 1.75+ (installez via [rustup](https://rustup.rs))
- Framework d'apprentissage profond Burn (ajouté comme dépendance Cargo)

### Installation
```bash
git clone https://github.com/AudricHarris/mamba-in-rust.git
cd mamba-in-rust
cargo build --release
```
