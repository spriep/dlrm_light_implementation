# Deep Learning Recommendation Model (DLRM) Implementation & DIEN Research

![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-ee4c2c) ![Docker](https://img.shields.io/badge/Docker-Container-2496ed) ![Status](https://img.shields.io/badge/Status-Educational-yellow)

## 📌 Project Overview

This repository hosts the implementation and analysis of state-of-the-art Deep Learning architectures for **Click-Through Rate (CTR) prediction**. The primary focus of this project is the deployment of Meta's **Deep Learning Recommendation Model (DLRM)**, transitioning from traditional collaborative filtering to advanced neural network–based approaches.

**Resource-Optimized Implementation**  
Given the typically high computational demands of industrial-scale recommendation systems, this implementation features a customized **"light" configuration**. This version has been specifically tuned to run efficiently on infrastructure with limited GPU memory, preventing common resource exhaustion (OOM) issues while maintaining architectural integrity for educational and testing purposes.

Additionally, this project includes a theoretical research component on the **Deep Interest Evolution Network (DIEN)**, exploring how user interest evolution can be modeled over time.

---

## 🏗️ DLRM Architecture

The core of this repository is the implementation of DLRM. This architecture is designed to handle the complexity of recommendation data by processing two distinct types of input features concurrently before combining them to predict user engagement.

### 1. Feature Processing
* **Sparse Features (Categorical):** Categorical inputs are processed using **Embedding Tables**. Each category is mapped to a dense representation vector, allowing the model to learn high-dimensional relationships between discrete entities.
* **Dense Features (Continuous):** Continuous numerical inputs are processed through the **Bottom MLP (Multi-Layer Perceptron)**, which transforms raw numerical features into a dense vector representation of the same dimension as the embedding vectors.

### 2. Interaction Layer
A defining feature of DLRM is how it computes feature combinations. The model performs **second-order interactions** between the dense representations (from the Bottom MLP) and the sparse embeddings via **dot products**.

* This method captures relationships between user/item attributes and continuous context, similar to matrix factorization but within a deep learning framework.
* Interactions are calculated between embeddings and processed dense features, but not among the outputs of the MLP layers themselves.

### 3. Prediction (Top MLP)
The resulting interaction vectors are concatenated with the original processed dense features and fed into the **Top MLP**, which acts as a classifier. The output is passed through a sigmoid function to generate a probability score (e.g., likelihood of a click).

### 4. Hybrid Parallelism Strategy
To handle the massive parameter size associated with recommendation embeddings, the architecture supports a hybrid parallelization strategy:

* **Model Parallelism:** Applied to embedding tables, distributing them across devices to alleviate memory pressure.
* **Data Parallelism:** Applied to the MLP components.
* **Butterfly Shuffle:** A custom communication operator used to synchronize and transfer embedding vectors efficiently during forward and backward passes across devices.

---

## 📊 Dataset

The model was trained and evaluated using the **Kaggle Display Advertising Dataset**, provided by Criteo Labs.

* **Source:** A portion of Criteo's ad traffic collected over 7 days.  
* **Size:** Approximately 45 million samples.  
* **Features:** 13 continuous features and 26 categorical features (39 total).  
* **Target:** Binary classification indicating whether an ad was clicked (1) or not (0).

---

## 📉 Training & Benchmarking

Training was performed inside a custom **Docker** environment to ensure reproducibility and enable the use of Linux-native libraries such as `fbgemm_gpu`.

### Training Performance
Even with resource-constrained settings and the optimized "light" configuration, the model demonstrated successful learning:

* **Loss:** Converged to approximately 0.675 in early epochs.  
* **Accuracy:** Achieved **90.00%** accuracy on the validation subset.

### Benchmarking
The repository includes scripts to produce detailed performance profiling outputs. These include `.prof` and `.json` files compatible with **PyTorch Profiler** and **Chrome Tracing** (`chrome://tracing`) for in-depth inspection of CPU and CUDA kernel execution times.

---

## 📑 Research Component: DIEN

Complementing the DLRM implementation, this project investigates the **Deep Interest Evolution Network (DIEN)**, which models **temporal dynamics of user interests**.

Key ideas in DIEN:

* **Interest Extractor Layer:** Based on GRUs, it captures latent interests from sequential user behavior and uses auxiliary loss to supervise the learning of intermediate interest states.
* **AUGRU (Attentional Update Gate GRU):** Introduces attention into the GRU update gate, allowing the model to highlight interest states most relevant to the target item. This increases CTR prediction accuracy by aligning user intent with the item being predicted.

---

*Project developed at Università degli Studi Roma Tre.*
