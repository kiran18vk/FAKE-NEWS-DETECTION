import streamlit as st
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_v3"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    tfidf = joblib.load(
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.pkl"
        )
    )

    logistic_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "logistic_regression.pkl"
        )
    )

    svm_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "linear_svm.pkl"
        )
    )

    random_forest = joblib.load(
        os.path.join(
            MODEL_DIR,
            "random_forest.pkl"
        )
    )

    svd = joblib.load(
        os.path.join(
            MODEL_DIR,
            "svd.pkl"
        )
    )

    return (
        tfidf,
        logistic_model,
        svm_model,
        random_forest,
        svd
    )


# ============================================================
# LOAD MODELS SAFELY
# ============================================================

try:

    (
        tfidf,
        logistic_model,
        svm_model,
        random_forest,
        svd
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "❌ Unable to load trained models."
    )

    st.code(str(e))

    st.info(
        "Make sure the models_v3 folder contains "
        "the trained model files."
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
    color: #172B4D;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
    color: #667085;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #D9DEE7;
    text-align: center;
    background-color: #FFFFFF;
    min-height: 180px;
}

.result-title {
    font-size: 20px;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 25px;
}

.result-label {
    font-size: 25px;
    font-weight: 700;
}

.final-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #D9DEE7;
    background-color: #F8FAFC;
}

.final-text {
    font-size: 30px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Fake News Classification'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# STOP IF MODELS FAILED
# ============================================================

if not models_loaded:
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    st.divider()

    st.subheader("📌 Input Mode")

    input_mode = st.radio(
        "Select how you want to provide the news:",
        [
            "Title Only",
            "Title + Article",
            "Article Only"
        ]
    )

    st.divider()

    st.subheader("📊 Model Information")

    st.write("**Feature Extraction:**")
    st.write("TF-IDF")

    st.write("**N-grams:**")
    st.write("Unigrams + Bigrams")

    st.write("**Models:**")
    st.write("• Logistic Regression")
    st.write("• Linear SVM")
    st.write("• Random Forest")

    st.write("**Random Forest:**")
    st.write("TF-IDF → SVD → Random Forest")

    st.write("**SVD Components:**")
    st.write("200")

    st.write("**Final Decision:**")
    st.write("Majority Voting")

    st.divider()

    st.subheader("🏷️ Label Mapping")

    st.success("**0 = REAL NEWS**")

    st.error("**1 = FAKE NEWS**")


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📝 Enter News Information")


title = ""
article = ""


# ============================================================
# TITLE ONLY
# ============================================================

if input_mode == "Title Only":

    title = st.text_input(
        "News Title",
        placeholder="Enter the news headline..."
    )

    st.info(
        "The prediction will be based only on the title."
    )


# ============================================================
# TITLE + ARTICLE
# ============================================================

elif input_mode == "Title + Article":

    title = st.text_input(
        "News Title",
        placeholder="Enter the news headline..."
    )

    article = st.text_area(
        "News Article",
        height=250,
        placeholder="Paste or type the full news article..."
    )

    st.info(
        "The prediction will use both the title and article."
    )


# ============================================================
# ARTICLE ONLY
# ============================================================

elif input_mode == "Article Only":

    article = st.text_area(
        "News Article",
        height=300,
        placeholder="Paste or type the full news article..."
    )

    st.info(
        "The prediction will be based only on the article."
    )


# ============================================================
# CREATE INPUT TEXT
# ============================================================

if input_mode == "Title Only":

    input_text = title.strip()

elif input_mode == "Title + Article":

    input_text = (
        title.strip()
        + " "
        + article.strip()
    ).strip()

else:

    input_text = article.strip()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(text):

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    X_tfidf = tfidf.transform(
        [text]
    )


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    lr_prediction = int(
        logistic_model.predict(
            X_tfidf
        )[0]
    )


    # --------------------------------------------------------
    # LINEAR SVM
    # --------------------------------------------------------

    svm_prediction = int(
        svm_model.predict(
            X_tfidf
        )[0]
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # TF-IDF → SVD → RANDOM FOREST
    # --------------------------------------------------------

    X_svd = svd.transform(
        X_tfidf
    )

    rf_prediction = int(
        random_forest.predict(
            X_svd
        )[0]
    )


    # --------------------------------------------------------
    # MAJORITY VOTING
    # --------------------------------------------------------

    predictions = [
        lr_prediction,
        svm_prediction,
        rf_prediction
    ]

    fake_votes = predictions.count(1)

    real_votes = predictions.count(0)


    if fake_votes > real_votes:

        final_prediction = 1
        winning_votes = fake_votes

    else:

        final_prediction = 0
        winning_votes = real_votes


    agreement = (
        winning_votes / 3
    ) * 100


    return (
        lr_prediction,
        svm_prediction,
        rf_prediction,
        final_prediction,
        winning_votes,
        agreement
    )


# ============================================================
# LABEL FUNCTIONS
# ============================================================

def label_text(prediction):

    if prediction == 1:
        return "FAKE"

    return "REAL"


def label_icon(prediction):

    if prediction == 1:
        return "🔴"

    return "🟢"


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔎 Check News",
    type="primary",
    use_container_width=True
)


# ============================================================
# VALIDATION + PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not input_text:

        st.warning(
            "⚠️ Please enter some news text before checking."
        )

        st.stop()


    # --------------------------------------------------------
    # MINIMUM TEXT
    # --------------------------------------------------------

    if len(input_text.strip()) < 5:

        st.warning(
            "⚠️ Please enter a little more text for prediction."
        )

        st.stop()


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        (
            lr_prediction,
            svm_prediction,
            rf_prediction,
            final_prediction,
            winning_votes,
            agreement
        ) = make_prediction(
            input_text
        )

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(str(e))

        st.stop()


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    lr_label = label_text(
        lr_prediction
    )

    svm_label = label_text(
        svm_prediction
    )

    rf_label = label_text(
        rf_prediction
    )

    final_label = label_text(
        final_prediction
    )

    final_icon = label_icon(
        final_prediction
    )


    # ========================================================
    # INDIVIDUAL MODEL PREDICTIONS
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Individual Model Predictions"
    )


    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    with col1:

        with st.container(border=True):

            st.markdown(
                "### Logistic Regression"
            )

            st.write("Prediction")

            if lr_prediction == 1:

                st.error(
                    "🔴 FAKE"
                )

            else:

                st.success(
                    "🟢 REAL"
                )


    # --------------------------------------------------------
    # LINEAR SVM
    # --------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.markdown(
                "### Linear SVM"
            )

            st.write("Prediction")

            if svm_prediction == 1:

                st.error(
                    "🔴 FAKE"
                )

            else:

                st.success(
                    "🟢 REAL"
                )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    with col3:

        with st.container(border=True):

            st.markdown(
                "### Random Forest"
            )

            st.write("Prediction")

            if rf_prediction == 1:

                st.error(
                    "🔴 FAKE"
                )

            else:

                st.success(
                    "🟢 REAL"
                )


    # ========================================================
    # MAJORITY VOTING
    # ========================================================

    st.divider()

    st.subheader(
        "🗳️ Majority Voting Result"
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    with st.container(border=True):

        if final_prediction == 1:

            st.error(
                "## 🔴 FINAL PREDICTION: FAKE NEWS"
            )

        else:

            st.success(
                "## 🟢 FINAL PREDICTION: REAL NEWS"
            )


        st.markdown(
            f"### Model Agreement: {winning_votes}/3"
        )

        st.progress(
            agreement / 100
        )

        st.write(
            f"**Agreement:** {agreement:.2f}%"
        )


    # ========================================================
    # VOTE BREAKDOWN
    # ========================================================

    st.write("")

    st.subheader(
        "📊 Vote Breakdown"
    )


    fake_votes = (
        (lr_prediction == 1)
        + (svm_prediction == 1)
        + (rf_prediction == 1)
    )


    real_votes = (
        (lr_prediction == 0)
        + (svm_prediction == 0)
        + (rf_prediction == 0)
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🔴 Fake Votes",
            f"{fake_votes}/3"
        )


    with col2:

        st.metric(
            "🟢 Real Votes",
            f"{real_votes}/3"
        )


    with col3:

        st.metric(
            "📊 Agreement",
            f"{agreement:.2f}%"
        )


    # ========================================================
    # DETAILED INFORMATION
    # ========================================================

    st.write("")

    with st.expander(
        "ℹ️ Prediction Information"
    ):

        st.write(
            f"**Input Mode:** {input_mode}"
        )

        st.write(
            "**Decision Method:** Majority Voting"
        )

        st.write(
            "**Models:** Logistic Regression, "
            "Linear SVM, Random Forest"
        )

        st.write(
            "**Feature Extraction:** TF-IDF"
        )

        st.write(
            "**N-grams:** Unigrams + Bigrams"
        )

        st.write(
            "**Random Forest:** TF-IDF → SVD → Random Forest"
        )

        st.write(
            "**SVD Components:** 200"
        )

        st.write(
            "**Label 0:** Real News"
        )

        st.write(
            "**Label 1:** Fake News"
        )

        st.caption(
            "Model Agreement represents the proportion "
            "of models supporting the final prediction. "
            "It is not a probability that the prediction "
            "is correct."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake News Detection System | "
    "TF-IDF + Unigrams/Bigrams + "
    "Logistic Regression + Linear SVM + "
    "Random Forest + Majority Voting"
)