from __future__ import annotations

import streamlit as st

from financial_sentiment.finbert import predict_finbert

st.set_page_config(page_title="Financial NLP Sentiment", layout="centered")
st.title("Financial NLP Sentiment")
st.caption("Classify short financial text with FinBERT.")

text = st.text_area(
    "Financial headline or market commentary",
    placeholder="Example: The company raised its full-year revenue guidance after strong demand.",
)

if st.button("Analyse sentiment", type="primary"):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        with st.spinner("Running model..."):
            result = predict_finbert(text)
        st.metric("Sentiment", str(result["label"]).title())
        st.metric("Confidence", f"{float(result['confidence']):.1%}")
        st.caption(f"Model: {result['model']}")
