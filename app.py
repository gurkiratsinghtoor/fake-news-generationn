"""Streamlit web app:  streamlit run app.py"""
import os, sys
import pandas as pd
import streamlit as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import load, predict
from api_generator import generate_samples

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Generator & Predictor")
st.caption("NLP + TF-IDF + Machine Learning")

# On Streamlit Cloud the key can be stored in "Secrets" (never in code / GitHub)
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

try:
    bundle = load()
except FileNotFoundError:
    st.error("Model not found. Run `python src/train.py` first."); st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 Predictor", "🧪 Synthetic News Generator", "📊 Model Report"])

with tab1:
    text = st.text_area("Paste a news headline / article", height=200)
    if st.button("Analyze") and text.strip():
        r = predict(text, bundle)
        (st.error if r["label"] == "FAKE" else st.success)(f"Prediction: **{r['label']}**")
        st.progress(r["fake_probability"], text=f"Fake probability: {r['fake_probability']:.1%}")
        if r["key_terms"]:
            st.write("Key terms behind the decision:")
            st.dataframe(pd.DataFrame(r["key_terms"], columns=["term", "influence (+ fake / - real)"]))

with tab2:
    st.write("Generates **fictional, labelled** samples (invented names/places) with the Gemini API "
             "(falls back to templates if `GEMINI_API_KEY` is not set) to test the model.")
    key_in = st.text_input("Gemini API key (or set GEMINI_API_KEY before starting the app)",
                           type="password", help="Free key: https://aistudio.google.com/apikey")
    topic = st.text_input("Topic", "local government and city services")
    n = st.slider("Samples per class", 1, 10, 3)
    if st.button("Generate"):
        with st.spinner("Generating..."):
            df, source, err = generate_samples(n, topic, key_in.strip() or None)
        if err:
            st.error(f"Gemini API was NOT used - showing template samples instead.\n\n{err}")
        else:
            st.success("Generated with Gemini API")
        st.caption(f"Source: {source}")
        df["model_says"] = [predict(t + " " + b, bundle)["label"] for t, b in zip(df.title, df.text)]
        df["true_label"] = df["label"].map({1: "FAKE", 0: "REAL"})
        st.metric("Model accuracy on generated samples", f"{(df.model_says == df.true_label).mean():.0%}")
        st.dataframe(df[["title", "text", "true_label", "model_says"]], use_container_width=True)

with tab3:
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    if os.path.exists(f"{rp}/model_comparison.csv"):
        st.dataframe(pd.read_csv(f"{rp}/model_comparison.csv"))
    for img in ("confusion_matrix.png", "top_words.png"):
        if os.path.exists(f"{rp}/{img}"):
            st.image(f"{rp}/{img}")
