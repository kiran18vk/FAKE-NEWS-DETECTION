# 📰 TruthLens - Fake News Detection System

TruthLens is an intelligent Machine Learning web application designed to detect and classify fake news articles using Natural Language Processing (NLP) techniques and ensemble machine learning classifiers.

---

## 🌟 Key Features

* **Multi-Model Intelligence:** Trained with Support Vector Machine (Linear SVM), Logistic Regression, and Random Forest (via TruncatedSVD), supported by a Majority Voting Ensemble.
* **Flexible Input Verification:** Evaluates articles across full text, headlines, and combined inputs.
* **Real-Time Interactive UI:** Built on Streamlit with interactive Plotly analytics (confusion matrices, accuracy gauges, precision/recall/F1 breakdowns).
* **High Accuracy Benchmark:** Achieves ~94.3% accuracy on standard test benchmarks using the WELFake dataset (~72,000 articles).

---

## 🏗️ Project Structure

```text
├── app_v3.py                   # Streamlit web application
├── prepare_dataset.py          # Data cleaning, deduplication & stratified splitting
├── train_model_v3.py           # Model training, vectorization & evaluation
├── train_model_v3_plots.py     # Training script with evaluation plots generation
├── requirements.txt            # Python dependencies
├── models_v3/                  # Serialized models, vectorizer & performance metrics
│   ├── linear_svm.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svd.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── verified_registry.json
│   ├── test_results.csv
│   └── plots/
└── prepared_data/              # Split dataset directory (train, val, test)
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone <YOUR_REPOSITORY_URL>
cd FNDS
pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit web dashboard:
```bash
streamlit run app_v3.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dataset Overview (WELFake Benchmark)

* **Dataset:** WELFake composite news benchmark (~72,134 original records).
* **Cleaning:** Deduplicated and filtered to 63,682 clean articles with balanced class distribution (~54.6% REAL, 45.4% FAKE).
* **Splits:** 70% Training (44,580), 15% Validation (9,551), and 15% Test (9,551).
* **Labels:** `0 = REAL`, `1 = FAKE`.

