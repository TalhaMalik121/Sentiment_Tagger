import re
import nltk
import torch
import pandas as pd
import streamlit as st
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import nltk

# Ensure punkt tokenizer is available
nltk.download("punkt")
nltk.download("punkt_tab")  # sometimes needed for newer NLTK versions


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Sentiment & Emotion Detector",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment & Emotion Detector")
st.caption("Paste a paragraph, click **Analyze**, and see overall + per-phrase sentiment & emotion.")

# -----------------------------
# NLTK setup (sentence splitter)
# -----------------------------
@st.cache_resource
def ensure_nltk_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

ensure_nltk_punkt()

# -----------------------------
# Device selection
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_IDX = 0 if torch.cuda.is_available() else -1  # for HF pipeline

# -----------------------------
# Load models (cached)
# -----------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "bhadresh-savani/bert-base-go-emotion"

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=DEVICE_IDX
    )

@st.cache_resource
def load_emotion_model():
    tok = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL).to(DEVICE)
    return tok, mdl

sentiment_pipe = load_sentiment_pipeline()
emotion_tok, emotion_mdl = load_emotion_model()

# -----------------------------
# Label prettifiers / helpers
# -----------------------------
def pretty_sentiment(label: str) -> str:
    mapping = {
        "LABEL_0": "😡 Negative",
        "LABEL_1": "😐 Neutral",
        "LABEL_2": "😄 Positive",
        "negative": "😡 Negative",
        "neutral": "😐 Neutral",
        "positive": "😄 Positive"
    }
    return mapping.get(label.lower(), label)

def predict_emotion(text: str) -> Tuple[str, float]:
    inputs = emotion_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = emotion_mdl(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    top_prob, top_class = torch.topk(probs, 1)
    label = emotion_mdl.config.id2label[top_class.item()]
    return label, float(top_prob.item())

def paragraph_to_phrases(paragraph: str) -> List[str]:
    sentences = nltk.sent_tokenize(paragraph)
    phrases = []
    for sentence in sentences:
        parts = re.split(r',|;|\band\b|\bbut\b|\bor\b|\bso\b|\byet\b', sentence, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        phrases.extend(parts)
    return phrases

# -----------------------------
# UI: Input
# -----------------------------
text = st.text_area(
    "Write Paragraph",
    value="",
    placeholder="Write Paragraph",
    height=180
)

analyze_clicked = st.button("🔍 Analyze")

# -----------------------------
# Processing
# -----------------------------
if analyze_clicked:
    if not text or not text.strip():
        st.warning("Please paste a paragraph first.")
    else:
        with st.spinner("Analyzing paragraph and phrases…"):
            # Overall paragraph sentiment
            s_res = sentiment_pipe(text)[0]
            sent_label = pretty_sentiment(s_res["label"])
            sent_score = float(s_res["score"])

            # Overall paragraph emotion
            emo_label, emo_score = predict_emotion(text)

            # Phrase-level
            phrases = paragraph_to_phrases(text)
            rows = []
            for phr in phrases:
                s = sentiment_pipe(phr)[0]
                p_label = pretty_sentiment(s["label"])
                p_score = float(s["score"])
                e_label, e_score = predict_emotion(phr)
                rows.append({
                    "Phrase": phr,
                    "Sentiment": p_label,
                    "Sentiment_Confidence": round(p_score, 3),
                    "Emotion": e_label,
                    "Emotion_Confidence": round(e_score, 3),
                })

            df = pd.DataFrame(rows)

        # -----------------------------
        # Results UI
        # -----------------------------
        st.subheader("📊 Overall (Paragraph-level)")
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**Sentiment:** {sent_label}  \n**Confidence:** {sent_score:.3f}")
        with c2:
            st.info(f"**Emotion:** {emo_label}  \n**Confidence:** {emo_score:.3f}")

        st.divider()

        st.subheader("✂ Phrase-level Analysis")
        if df.empty:
            st.write("_No phrases detected; try a longer paragraph or different punctuation._")
        else:
            st.dataframe(df, use_container_width=True)

            # Download CSV
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download phrase results (CSV)",
                data=csv_bytes,
                file_name="phrase_results.csv",
                mime="text/csv"
            )

# Footer tip
st.caption("Tip: First run may take a bit while models download. Subsequent runs are much faster.")

