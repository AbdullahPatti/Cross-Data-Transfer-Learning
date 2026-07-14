# Cross-Domain Transfer Learning for Text Classification

A machine learning project exploring **cross-domain and cross-lingual transfer learning** using **XLM-RoBERTa** across three NLP tasks: Urdu fake news detection, English fake news detection, and sarcasm/irony detection. Each task benchmarks transformer fine-tuning against classical ML baselines (TF-IDF + Logistic Regression & SVM) and evaluates how well models generalize across datasets.

---

## Project Overview

This project investigates whether a model trained on one dataset can generalize to another — both within a task (cross-domain) and across languages (cross-lingual). Three independent experiments are conducted:

| Task | Language | Notebooks |
|---|---|---|
| Urdu Fake News Detection | Urdu | `UrduFakeNews.ipynb`, `UrduFakeNews2.ipynb` |
| English Fake News Detection | English | `WelFake.ipynb` |
| Sarcasm / Irony Detection | English | `Sarcasm.ipynb` |

---

## Notebooks

| Notebook | Description |
|---|---|
| `UrduFakeNews.ipynb` | EDA, preprocessing, dataset merging, and XLM-RoBERTa fine-tuning on the Ax-to-Grind dataset (Dataset A) |
| `UrduFakeNews2.ipynb` | Cross-domain experiments, fine-tuning on Notri-Fact (Dataset B), TF-IDF + ML baselines, word frequency & label distribution analysis |
| `WelFake.ipynb` | English fake news detection using the WELFake dataset; in-domain and cross-domain experiments with XLM-RoBERTa and ML baselines |
| `Sarcasm.ipynb` | Sarcasm/irony detection using TweetEval (Dataset A) and a multi-source sarcasm corpus (Dataset B); cross-domain experiments |

---

## Datasets

### Urdu Fake News
- **Dataset A — Ax-to-Grind** (`Combined.csv`): Urdu news articles labeled as `FAKE` or `TRUE`
- **Dataset B — Notri-Fact** (`Notri-Fact_Real_Unreal_Urdu_NEWS.xlsx`): Urdu news labeled as `Real` or `Unreal`, including headline and body text

### English Fake News
- **WELFake** (`WELFake_Dataset.csv`): Large-scale English fake/real news dataset combining multiple sources

### Sarcasm / Irony
- **Dataset A — TweetEval** (`cardiffnlp/tweet_eval` irony split, loaded via HuggingFace Datasets): Twitter irony detection
- **Dataset B — Multi-source sarcasm corpus**: Combines `GEN-sarc-notsarc.csv`, `HYP-sarc-notsarc.csv`, and `RQ-sarc-notsarc.csv`

---

## Methodology

### Phase 1 — Data Preprocessing & EDA
- Label standardization and binary mapping
- Duplicate removal and dataset merging
- Text length and word count distribution analysis per class
- Language-specific text cleaning (Urdu stopword removal, Arabic/Urdu rendering; English tokenization)

### Phase 2 — Traditional ML Baselines
- TF-IDF vectorization (top 10,000 features)
- Logistic Regression
- Support Vector Machine (SVM) with linear kernel
- Confusion matrices for visual evaluation

### Phase 3 — Transformer Fine-tuning
- Model: `xlm-roberta-base` (multilingual, handles both Urdu and English)
- Tokenization with `max_length=256`
- Training via HuggingFace `Trainer` API
- Metrics: Accuracy & F1-score
- 3 epochs, learning rate `2e-5`, weight decay `0.01`

### Phase 4 — Cross-Domain Experiments

Each task runs four experiments:

| Experiment | Train On | Test On |
|---|---|---|
| Exp 1 | Dataset A | Dataset A (in-domain) |
| Exp 2 | Dataset B | Dataset B (in-domain) |
| Exp 3 | Dataset A | Dataset B (cross-domain) |
| Exp 4 | Dataset B | Dataset A (cross-domain) |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | XLM-RoBERTa, TF-IDF, NLTK |
| ML | scikit-learn (Logistic Regression, SVM) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, arabic_reshaper, python-bidi |
| Platform | Google Colab |

---

## Installation

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy torch matplotlib seaborn scikit-learn \
            transformers datasets evaluate accelerate openpyxl \
            nltk arabic-reshaper python-bidi
```

---

## How to Run

1. Clone this repository
2. For Urdu notebooks: upload datasets to Google Drive under `MyDrive/UrduFakeNews/`
3. Open the relevant notebook in Google Colab and run all cells

> A GPU runtime is strongly recommended for transformer fine-tuning.

---

## Project Structure

```
Cross-Data-Transfer-Learning/
│
├── IPYNB Files/
│   ├── UrduFakeNews.ipynb       # Urdu fake news: EDA, preprocessing, Exp 1
│   ├── UrduFakeNews2.ipynb      # Urdu fake news: cross-domain, baselines
│   ├── WelFake.ipynb            # English fake news: all experiments
│   └── Sarcasm.ipynb            # Sarcasm/irony detection: all experiments
│
├── Code Files/
│   ├── urdufakenews.py          # Python export of UrduFakeNews.ipynb
│   └── urdufakenews2.py         # Python export of UrduFakeNews2.ipynb
│
├── Datasets/
│   ├── Combined .csv
│   ├── Fake.csv
│   ├── True.csv
│   ├── WELFake_Dataset.csv
│   ├── GEN-sarc-notsarc.csv
│   ├── HYP-sarc-notsarc.csv
│   ├── RQ-sarc-notsarc.csv
│   └── Notri-Fact_Real_Unreal_Urdu_NEWS.xlsx
│
├── Images/                      # Result plots and figures
├── requirements.txt
└── README.md
```

---

## Future Work

- [ ] Experiment with Urdu-specific models (e.g., `urduhack`, `bert-base-multilingual-cased`)
- [ ] Add headline-body consistency features for fake news detection
- [ ] Explore cross-lingual transfer between Urdu and English tasks
- [ ] Deploy as a web app for real-time classification
- [ ] Expand to more datasets for better generalization benchmarks

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
