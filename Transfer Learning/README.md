# 🕵️ Urdu Fake News Detection using XLM-RoBERTa

A machine learning project for detecting fake news in the **Urdu language**, combining classical NLP approaches with state-of-the-art transformer models. This project explores both in-domain and cross-domain generalization across two distinct Urdu news datasets.

---

## 📌 Project Overview

Misinformation in low-resource languages like Urdu is a growing concern. This project tackles that challenge by fine-tuning **XLM-RoBERTa** on Urdu news data and benchmarking it against traditional ML baselines (Logistic Regression & SVM with TF-IDF). A key focus is **cross-domain robustness** — evaluating whether a model trained on one dataset can generalize to another.

---

## 📂 Notebooks

| Notebook | Description |
|---|---|
| `UrduFakeNews.ipynb` | Data loading, EDA, preprocessing, dataset merging, and initial XLM-RoBERTa fine-tuning on Dataset A |
| `UrduFakeNews2.ipynb` | Cross-domain experiments, fine-tuning on Dataset B, TF-IDF + ML baselines, word frequency & label distribution analysis |

---

## 🗃️ Datasets

Two Urdu news datasets are used:

- **Dataset A — Ax-to-Grind** (`Combined.csv`): A CSV dataset with Urdu news articles labeled as `FAKE` or `TRUE`.
- **Dataset B — Notri-Fact** (`Notri-Fact_Real_Unreal_Urdu_NEWS.xlsx`): An Excel dataset with Urdu news labeled as `Real` or `Unreal`, including headline and body text.

Both datasets are standardized, combined, deduplicated, and analyzed before modeling.

---

## ⚙️ Methodology

### 🔹 Phase 1 — Data Preprocessing & EDA
- Label standardization and mapping to binary (0 = Fake, 1 = Real)
- Duplicate removal and dataset merging
- Word count distribution analysis per class
- Urdu-specific text cleaning (stopword removal, punctuation, digit normalization)
- Word frequency analysis with proper Arabic/Urdu rendering

### 🔹 Phase 2 — Traditional ML Baselines
- TF-IDF vectorization (top 10,000 features)
- Logistic Regression
- Support Vector Machine (SVM) with linear kernel
- Confusion matrices for visual evaluation

### 🔹 Phase 3 — Transformer Fine-tuning
- Model: `xlm-roberta-base` (multilingual, handles Urdu well)
- Tokenization with `max_length=256`
- Training using HuggingFace `Trainer` API
- Metrics: Accuracy & F1-score
- 3 epochs, learning rate `2e-5`, weight decay `0.01`

### 🔹 Phase 4 — Cross-Domain Experiments
| Experiment | Train On | Test On |
|---|---|---|
| Exp 1 | Dataset A (Ax-to-Grind) | Dataset A |
| Exp 2 | Dataset B (Notri-Fact) | Dataset B |
| Exp 3 | Dataset A | Dataset B |
| Exp 4 | Dataset B | Dataset A |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | XLM-RoBERTa, TF-IDF, NLTK |
| ML | scikit-learn (LR, SVM) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, arabic_reshaper, python-bidi |
| Platform | Google Colab |

---

## 📦 Installation

```bash
pip install pandas numpy torch matplotlib seaborn scikit-learn \
            transformers datasets evaluate accelerate openpyxl \
            nltk arabic_reshaper python-bidi
```

---

## 🚀 How to Run

1. Clone this repository
2. Upload your datasets to Google Drive under `MyDrive/UrduFakeNews/`
3. Open `UrduFakeNews.ipynb` in Google Colab and run all cells (data prep + Exp 1 training)
4. Open `UrduFakeNews2.ipynb` in Google Colab for cross-domain experiments and ML baselines

> 💡 A GPU runtime is strongly recommended for transformer fine-tuning.

---

## 📁 Project Structure

```
urdu-fake-news-detection/
│
├── Code Files/
│   ├── UrduFakeNews.py       # Notebook 1: EDA, preprocessing, Exp 1
│   └── UrduFakeNews2.py       # Python source code
│
├── Jupyter/
│   ├── UrduFakeNews.ipynb         # Notebook 1: EDA, preprocessing, Exp 1
│   └── UrduFakeNews2.ipynb        # Notebook 2: Cross-domain, baselines, analysis
│
└── README.md
```

---

## 🔮 Future Work

- [ ] Experiment with Urdu-specific models (e.g., `urduhack`, `bert-base-multilingual-cased`)
- [ ] Add headline-body consistency features
- [ ] Deploy as a web app for real-time fake news detection
- [ ] Expand to more Urdu news sources for better generalization

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

