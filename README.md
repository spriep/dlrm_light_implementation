# Deep Learning Recommendation Model (DLRM) Implementation & DIEN Research

![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-ee4c2c) ![Docker](https://img.shields.io/badge/Docker-Container-2496ed) ![Status](https://img.shields.io/badge/Status-Educational-yellow)

## 📌 Project Overview

[cite_start]This repository hosts the implementation and analysis of state-of-the-art Deep Learning architectures for **Click-Through Rate (CTR) prediction**[cite: 218]. [cite_start]The primary focus of this project is the deployment of Meta's **Deep Learning Recommendation Model (DLRM)**, transitioning from traditional collaborative filtering to advanced neural network-based approaches[cite: 3, 21].

**Resource-Optimized Implementation**
[cite_start]Given the typically high computational demands of industrial-scale recommendation systems, this implementation features a customized **"light" configuration**[cite: 113]. [cite_start]This version has been specifically tuned to run efficiently on infrastructure with limited GPU memory, preventing common resource exhaustion (OOM) issues while maintaining architectural integrity for educational and testing purposes[cite: 112].

[cite_start]Additionally, this project includes a theoretical research component on the **Deep Interest Evolution Network (DIEN)**, exploring how user interest evolution can be modeled over time[cite: 207, 228].

---

## 🏗️ DLRM Architecture

The core of this repository is the implementation of DLRM. [cite_start]This architecture is designed to handle the complexity of recommendation data by processing two distinct types of input features concurrently before combining them to predict user engagement[cite: 30].

### 1. Feature Processing
* **Sparse Features (Categorical):** Categorical inputs are processed using **Embedding Tables**. [cite_start]Each category is mapped to a dense representation vector in an abstract space, allowing the model to learn high-dimensional relationships between discrete entities[cite: 24, 25, 33].
* **Dense Features (Continuous):** Continuous numerical inputs are processed through the **Bottom MLP (Multi-Layer Perceptron)**. [cite_start]This neural network transforms raw numerical features into a dense vector representation of the same length as the embedding vectors[cite: 29, 33].

### 2. Interaction Layer
The defining feature of DLRM is how it handles feature combinations. [cite_start]The model computes **Second-Order Interactions** between the dense representations (from the Bottom MLP) and the sparse embeddings via **Dot Products**[cite: 34].
* [cite_start]This approach explicitly captures the relationship between user/item attributes and continuous context, similar to Matrix Factorization but within a deep learning framework[cite: 27].
* [cite_start]Interactions are calculated between embeddings and processed dense features, but not among the outputs of the MLP themselves[cite: 35].

### 3. Prediction (Top MLP)
The resulting interaction vectors are concatenated with the original processed dense features and fed into the **Top MLP**. [cite_start]This final neural network acts as a classifier, passing the output through a sigmoid function to generate a probability score (e.g., the likelihood of a click)[cite: 37].

### 4. Hybrid Parallelism Strategy
[cite_start]To handle the massive parameter size typical of recommendation embeddings, the underlying architecture supports a hybrid parallelization strategy that frameworks like standard PyTorch do not natively support[cite: 42, 48].
* [cite_start]**Model Parallelism:** Applied to embedding tables (distributed across devices) to alleviate memory bottlenecks[cite: 46].
* [cite_start]**Data Parallelism:** Applied to the MLP components[cite: 46].
* [cite_start]**Butterfly Shuffle:** A custom communication operator used to synchronize and transfer embedding vectors between devices efficiently during the forward and backward passes[cite: 52].

---

## 📊 Dataset

[cite_start]The model was trained and benchmarked using the **Kaggle Display Advertising Dataset**, provided by Criteo Labs[cite: 76].

* [cite_start]**Source:** A portion of Criteo's traffic over a period of 7 days[cite: 77].
* [cite_start]**Volume:** Approximately 45 million entries[cite: 78].
* [cite_start]**Features:** Each entry contains 39 features, consisting of 13 continuous features and 26 categorical features[cite: 78].
* [cite_start]**Target:** Binary classification indicating whether an ad was clicked (1) or not (0)[cite: 77].

---

## 📉 Training & Benchmarking

[cite_start]The training process was executed within a custom **Docker** environment to ensure reproducibility and access to Linux-native libraries like `fbgemm_gpu`[cite: 65, 67].

### Training Performance
Despite resource constraints and the use of the optimized "light" parameters, the model successfully captured meaningful patterns from the dataset:
* [cite_start]**Loss:** The model converged to a loss of approximately 0.675 in initial epochs[cite: 137].
* [cite_start]**Accuracy:** Achieved a testing accuracy of **90.00%** on the validation subset[cite: 138].

### Benchmarking
The repository includes scripts to generate detailed performance profiling. [cite_start]Outputs include `.prof` and `.json` files compatible with **PyTorch Profiler** and **Chrome Tracing** (`chrome://tracing`), allowing for granular analysis of CPU and CUDA kernel execution times[cite: 144, 145, 146].

---

## 📑 Research Component: DIEN

[cite_start]Complementing the practical implementation of DLRM, this project investigates the **Deep Interest Evolution Network (DIEN)**[cite: 208].

While DLRM focuses on feature interaction, DIEN addresses the temporal dynamic of user behavior. [cite_start]It posits that user interests are not static but evolve over time[cite: 228]. The architecture introduces:
* [cite_start]**Interest Extractor Layer:** Uses GRUs to capture latent interests from behavior sequences, supervised by an auxiliary loss function[cite: 235, 236].
* [cite_start]**AUGRU (Attentional Update Gate GRU):** A novel mechanism that modulates the update gate based on the relevance of the target item[cite: 244]. [cite_start]This allows the model to emphasize relevant interests relative to a specific target ad, improving CTR prediction accuracy[cite: 246].

---

*Project developed at Università degli Studi Roma Tre.*
