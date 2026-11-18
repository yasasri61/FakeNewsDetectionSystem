from pathlib import Path
import json
import joblib
import streamlit as st

MODEL_PATH = Path("model/fake_news_pipeline.joblib")
METRICS_PATH = Path("model/metrics.json")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📰 AI Fake News Detection")
st.caption("A machine-learning mini project using TF-IDF and Logistic Regression.")

if not MODEL_PATH.exists():
    st.error("The trained model was not found.")
    st.info("First run: `python train.py` — then run: `streamlit run app.py`.")
    st.stop()

model = load_model()

st.warning(
    "Educational demonstration only. This model learns patterns in its training "
    "dataset; it does not fact-check claims or prove whether an article is true."
)

article = st.text_area(
    "Paste a news headline or article:",
    height=220,
    placeholder="Enter the news text here..."
)

if st.button("Analyze news", type="primary", use_container_width=True):
    if len(article.strip()) < 30:
        st.warning("Please enter at least 30 characters for a more useful prediction.")
    else:
        prediction = int(model.predict([article])[0])
        probabilities = model.predict_proba([article])[0]
        fake_probability = float(probabilities[0])
        real_probability = float(probabilities[1])

        if prediction == 1:
            st.success("Model prediction: REAL NEWS")
        else:
            st.error("Model prediction: FAKE NEWS")

        st.subheader("Model scores")
        col1, col2 = st.columns(2)
        col1.metric("Fake score", f"{fake_probability * 100:.1f}%")
        col2.metric("Real score", f"{real_probability * 100:.1f}%")
        st.progress(real_probability)

        st.caption(
            "Scores are model confidence estimates, not verified probabilities "
            "that the story is factually true."
        )

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    st.divider()
    st.subheader("Test-set performance")
    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    st.caption(f"Evaluated on {metrics['test_rows']:,} held-out articles.")
    st.write("Confusion matrix (rows = actual, columns = predicted; FAKE, REAL):")
    st.table(metrics["confusion_matrix"])
