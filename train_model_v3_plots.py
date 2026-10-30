import os
import time
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt



# ============================================================
# FAKE NEWS DETECTION - TRAINING V3
#
# Input modes:
#   1. Title Only
#   2. Article Only
#   3. Title + Article
#
# Label:
#   0 = REAL
#   1 = FAKE
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

TRAIN_FILE = "prepared_data/train.csv"
VAL_FILE = "prepared_data/validation.csv"
TEST_FILE = "prepared_data/test.csv"

MODEL_DIR = "models_v3"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Original training articles   : {len(train_df):,}")
print(f"Original validation articles : {len(val_df):,}")
print(f"Original test articles       : {len(test_df):,}")


# ============================================================
# 3. CREATE THREE INPUT MODES
# ============================================================

def create_three_modes(df):

    rows = []

    for _, row in df.iterrows():

        title = str(row["title"])
        article = str(row["text"])
        label = int(row["label"])

        # ----------------------------------------------------
        # MODE 1: TITLE ONLY
        # ----------------------------------------------------

        if title.strip():

            rows.append({
                "input_text": title.strip(),
                "input_mode": "Title Only",
                "label": label
            })

        # ----------------------------------------------------
        # MODE 2: ARTICLE ONLY
        # ----------------------------------------------------

        if article.strip():

            rows.append({
                "input_text": article.strip(),
                "input_mode": "Article Only",
                "label": label
            })

        # ----------------------------------------------------
        # MODE 3: TITLE + ARTICLE
        # ----------------------------------------------------

        combined = (
            title.strip()
            + " "
            + article.strip()
        ).strip()

        if combined:

            rows.append({
                "input_text": combined,
                "input_mode": "Title + Article",
                "label": label
            })

    return pd.DataFrame(rows)


# Create three versions
train_expanded = create_three_modes(train_df)
val_expanded = create_three_modes(val_df)
test_expanded = create_three_modes(test_df)


print("\nExpanded dataset sizes:")

print(
    f"Training examples   : "
    f"{len(train_expanded):,}"
)

print(
    f"Validation examples : "
    f"{len(val_expanded):,}"
)

print(
    f"Testing examples    : "
    f"{len(test_expanded):,}"
)


print("\nTraining input modes:")

print(
    train_expanded["input_mode"]
    .value_counts()
)


# ============================================================
# 4. TEXT AND LABELS
# ============================================================

X_train_text = train_expanded["input_text"]
X_val_text = val_expanded["input_text"]
X_test_text = test_expanded["input_text"]

y_train = train_expanded["label"]
y_val = val_expanded["label"]
y_test = test_expanded["label"]


# ============================================================
# 5. TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("CREATING TF-IDF FEATURES")
print("=" * 70)

start_time = time.time()

vectorizer = TfidfVectorizer(

    lowercase=True,

    strip_accents="unicode",

    # Unigrams + Bigrams
    ngram_range=(1, 2),

    min_df=2,

    max_features=150000,

    sublinear_tf=True
)


# IMPORTANT:
# Fit ONLY on training data

X_train = vectorizer.fit_transform(
    X_train_text
)

X_val = vectorizer.transform(
    X_val_text
)

X_test = vectorizer.transform(
    X_test_text
)


print(
    f"Vocabulary size: "
    f"{len(vectorizer.vocabulary_):,}"
)

print(
    f"Training matrix: "
    f"{X_train.shape}"
)

print(
    f"Validation matrix: "
    f"{X_val.shape}"
)

print(
    f"Test matrix: "
    f"{X_test.shape}"
)

print(
    f"TF-IDF time: "
    f"{time.time() - start_time:.2f} seconds"
)


# Save vectorizer

joblib.dump(
    vectorizer,
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{f1 * 100:.2f}%"
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

start_time = time.time()

lr_model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    solver="liblinear",
    class_weight=None
)

lr_model.fit(
    X_train,
    y_train
)

print(
    f"Training time: "
    f"{time.time() - start_time:.2f} seconds"
)


lr_val_result = evaluate_model(
    "Logistic Regression - Validation",
    lr_model,
    X_val,
    y_val
)

lr_test_result = evaluate_model(
    "Logistic Regression - Test",
    lr_model,
    X_test,
    y_test
)


joblib.dump(
    lr_model,
    os.path.join(
        MODEL_DIR,
        "logistic_regression.pkl"
    )
)


# ============================================================
# 8. LINEAR SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LINEAR SVM")
print("=" * 70)

start_time = time.time()

svm_model = LinearSVC(
    C=1.5,
    class_weight=None,
    max_iter=5000
)

svm_model.fit(
    X_train,
    y_train
)

print(
    f"Training time: "
    f"{time.time() - start_time:.2f} seconds"
)


svm_val_result = evaluate_model(
    "Linear SVM - Validation",
    svm_model,
    X_val,
    y_val
)

svm_test_result = evaluate_model(
    "Linear SVM - Test",
    svm_model,
    X_test,
    y_test
)


joblib.dump(
    svm_model,
    os.path.join(
        MODEL_DIR,
        "linear_svm.pkl"
    )
)


# ============================================================
# 9. SVD
# ============================================================

print("\n" + "=" * 70)
print("CREATING SVD FEATURES")
print("=" * 70)

start_time = time.time()

svd = TruncatedSVD(
    n_components=200,
    random_state=42
)

X_train_svd = svd.fit_transform(
    X_train
)

X_val_svd = svd.transform(
    X_val
)

X_test_svd = svd.transform(
    X_test
)


print(
    f"Training SVD shape: "
    f"{X_train_svd.shape}"
)

print(
    f"Validation SVD shape: "
    f"{X_val_svd.shape}"
)

print(
    f"Test SVD shape: "
    f"{X_test_svd.shape}"
)

print(
    f"SVD time: "
    f"{time.time() - start_time:.2f} seconds"
)


joblib.dump(
    svd,
    os.path.join(
        MODEL_DIR,
        "svd.pkl"
    )
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight=None,
    max_features="sqrt"
)

rf_model.fit(
    X_train_svd,
    y_train
)

print(
    f"Training time: "
    f"{time.time() - start_time:.2f} seconds"
)


rf_val_result = evaluate_model(
    "Random Forest - Validation",
    rf_model,
    X_val_svd,
    y_val
)

rf_test_result = evaluate_model(
    "Random Forest - Test",
    rf_model,
    X_test_svd,
    y_test
)


joblib.dump(
    rf_model,
    os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )
)


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

validation_results = pd.DataFrame([
    lr_val_result,
    svm_val_result,
    rf_val_result
])

test_results = pd.DataFrame([
    lr_test_result,
    svm_test_result,
    rf_test_result
])


print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print(
    validation_results[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

print(
    test_results[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    ].to_string(index=False)
)



# ============================================================
# 12. CONFUSION MATRICES + PLOTS
# ============================================================

PLOT_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Test-set predictions
lr_test_pred = lr_model.predict(X_test)
svm_test_pred = svm_model.predict(X_test)
rf_test_pred = rf_model.predict(X_test_svd)

# 3-model hard majority voting
majority_test_pred = (
    lr_test_pred.astype(int)
    + svm_test_pred.astype(int)
    + rf_test_pred.astype(int)
) >= 2
majority_test_pred = majority_test_pred.astype(int)

# Majority voting metrics
majority_result = {
    "Model": "Majority Voting",
    "Accuracy": accuracy_score(y_test, majority_test_pred),
    "Precision": precision_score(y_test, majority_test_pred, zero_division=0),
    "Recall": recall_score(y_test, majority_test_pred, zero_division=0),
    "F1": f1_score(y_test, majority_test_pred, zero_division=0)
}

test_results = pd.concat(
    [test_results, pd.DataFrame([majority_result])],
    ignore_index=True
)

# ------------------------------------------------------------
# CONFUSION MATRICES
# ------------------------------------------------------------

cm_models = {
    "Logistic Regression": lr_test_pred,
    "Linear SVM": svm_test_pred,
    "Random Forest": rf_test_pred,
    "Majority Voting": majority_test_pred
}

for model_name, predictions in cm_models.items():

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["REAL", "FAKE"]
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{model_name} - Test Confusion Matrix",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = list(range(len(test_results)))
width = 0.18

for i, metric in enumerate(metrics):

    values = test_results[metric] * 100

    ax.bar(
        [v + (i - 1.5) * width for v in x],
        values,
        width=width,
        label=metric
    )

ax.set_xticks(x)
ax.set_xticklabels(
    test_results["Model"],
    rotation=15,
    ha="right"
)

ax.set_ylabel("Score (%)")
ax.set_title(
    "Test Set Model Performance Comparison",
    fontsize=14,
    fontweight="bold"
)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "model_performance_comparison.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# TEST DATASET CLASS DISTRIBUTION
# ------------------------------------------------------------

label_counts = pd.Series(y_test).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(
    ["REAL", "FAKE"],
    [
        label_counts.get(0, 0),
        label_counts.get(1, 0)
    ]
)

ax.set_title(
    "Test Dataset Class Distribution",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("News Class")
ax.set_ylabel("Number of Articles")

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height()):,}",
        ha="center",
        va="bottom"
    )

ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "test_class_distribution.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# Save confusion matrices for all models
for model_name, predictions in cm_models.items():
    cm_val = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
    safe_name = model_name.lower().replace(" ", "_")
    pd.DataFrame(
        cm_val,
        index=["Actual REAL", "Actual FAKE"],
        columns=["Predicted REAL", "Predicted FAKE"]
    ).to_csv(
        os.path.join(
            MODEL_DIR,
            f"{safe_name}_confusion_matrix.csv"
        )
    )

test_results.to_csv(
    os.path.join(
        MODEL_DIR,
        "test_results_with_majority_voting.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("TEST CONFUSION MATRICES")
print("=" * 80)

for model_name, predictions in cm_models.items():
    print(f"\n{model_name}:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print(f"\nPlots saved to: {PLOT_DIR}")

# ============================================================
# 13. SAVE RESULTS
# ============================================================

validation_results.to_csv(
    os.path.join(
        MODEL_DIR,
        "validation_results.csv"
    ),
    index=False
)

test_results.to_csv(
    os.path.join(
        MODEL_DIR,
        "test_results.csv"
    ),
    index=False
)


# ============================================================
# 14. SAVE INFORMATION
# ============================================================

model_info = {

    "label_mapping": {
        "0": "REAL",
        "1": "FAKE"
    },

    "input_modes": [
        "Title Only",
        "Article Only",
        "Title + Article"
    ],

    "tfidf_ngram_range": "(1, 2)",

    "tfidf_max_features": 150000,

    "svd_components": 200,

    "models": [
        "Logistic Regression",
        "Linear SVM",
        "Random Forest"
    ],

    "decision_method": "Majority Voting"
}


joblib.dump(
    model_info,
    os.path.join(
        MODEL_DIR,
        "model_info.pkl"
    )
)


# ============================================================
# 15. COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nLabel mapping:")
print("0 = REAL")
print("1 = FAKE")

print("\nInput modes trained:")
print("1. Title Only")
print("2. Article Only")
print("3. Title + Article")

print("\nSaved files:")

print(
    "  models_v3/tfidf_vectorizer.pkl"
)

print(
    "  models_v3/logistic_regression.pkl"
)

print(
    "  models_v3/linear_svm.pkl"
)

print(
    "  models_v3/svd.pkl"
)

print(
    "  models_v3/random_forest.pkl"
)

print(
    "  models_v3/validation_results.csv"
)

print(
    "  models_v3/test_results.csv"
)

print(
    "  models_v3/model_info.pkl"
)