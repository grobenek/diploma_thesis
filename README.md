# Sentiment Analysis and Opinion Detection using Deep Neural Networks

This repository contains code for building and evaluating deep learning models for sentiment classification and opinion detection in unstructured text data.

---

## 📘 Project Overview

The goal of this project is to explore and evaluate different techniques for improving the performance of sentiment and opinion classification models. The focus is on the following four approaches:

- **Combining multiple word embedding models** (GloVe, Word2Vec, FastText)
- **Using Focal Loss** to address class imbalance
- **Standard Data Distillation (SDD)**: semi-supervised learning with pseudo-labeling
- **Filtered Data Distillation (FDD)**: enhanced SDD using confidence-based filtering

A custom experimental pipeline was implemented to allow systematic experimentation, hyperparameter tuning, and evaluation using F1-score.

---

## 🧠 Model Architecture

Each model is a feed-forward neural network with:

- Input shape: `(26, 300)` or `(78, 300)` depending on embedding strategy
- Flatten layer
- Two hidden Dense layers with ReLU activation
- Dropout and BatchNormalization layers
- Output layer with:
  - **Softmax** activation (for 3-class sentiment classification)
  - **Sigmoid** activation (for binary opinion detection)

---

## 🧪 Experiments

Four main experiments were conducted, each addressing a specific research question:

| Experiment | Goal | Technique |
|------------|------|-----------|
| **RQ1** | Improve performance using richer input representations | Embedding combination |
| **RQ2** | Handle class imbalance | Focal Loss |
| **RQ3** | Use unlabeled data effectively | Standard Data Distillation |
| **RQ4** | Improve robustness and reduce noise | Filtered Data Distillation |

All models were evaluated using 5-fold cross-validation, and statistical significance of results was assessed using the Mann-Whitney U test.

---

## 📊 Results Summary

| Experiment | Sentiment F1 | Opinion F1 | Result |
|------------|---------------|-------------|--------|
| Baseline   | 59.97%        | 62.20%      | -      |
| RQ1        | 62.88%        | 64.89%      | ✅ Statistically significant |
| RQ2        | 60.18%        | -           | ❌ Not significant |
| RQ3        | 61.96%        | 62.73%      | ✅ Partial improvement |
| RQ4        | 62.59%        | 64.89%      | ✅ Strongest overall result |

---

## 🔬 Methodology Highlights

- **Embedding Models:** GloVe, Word2Vec, FastText
- **Evaluation Metric:** F1-score
- **Validation:** 5-fold cross-validation (10 runs)
- **Hyperparameter Optimization:** GridSearch
- **Statistical Testing:** Mann-Whitney U test

---

## 🧩 Use Cases

- Automated analysis of customer feedback
- Opinion monitoring on social media
- Extracting sentiment trends for products or services

---

## 🛠 Technologies

- Python
- TensorFlow / Keras
- NumPy, Pandas, Scikit-learn, Matplotlib
