
# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy
from scipy.spatial import distance

st.title("🎯 Big Win Oz Lotto Predictor")

NUMBERS_RANGE = list(range(1, 48))
NUM_MAIN = 7
NUM_SUPP = 2
NUM_SETS = 10

st.markdown("### 📂 Upload Historical Draws")
uploaded_csv = st.file_uploader("Upload CSV with 7 main numbers per row", type="csv")

if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.success("✅ File uploaded")
    st.dataframe(df.head())

    all_numbers = df.iloc[:, :7].values.flatten()
    historical_freq = pd.Series(all_numbers).value_counts().sort_index()
    for num in NUMBERS_RANGE:
        if num not in historical_freq:
            historical_freq[num] = 0
    historical_freq = historical_freq.sort_index()
    st.bar_chart(historical_freq)

    def generate_predictions():
        predictions = []
        hot_scores = (historical_freq >= np.percentile(historical_freq, 75)).astype(int)
        cold_scores = (historical_freq <= np.percentile(historical_freq, 25)).astype(int)

        for _ in range(NUM_SETS):
            scores = historical_freq.copy().astype(float)
            noise = pd.Series(np.random.randn(47), index=np.arange(1, 48))  # 🔧 FIX HERE
            scores += noise
            scores += hot_scores * 1.5
            scores -= cold_scores * 0.5
            probs = scores / scores.sum()
            mains = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
            remaining = list(set(NUMBERS_RANGE) - set(mains))
            supps = np.random.choice(remaining, size=NUM_SUPP, replace=False)
            predictions.append(sorted(mains) + sorted(supps))
        return predictions

    st.markdown("### 🔮 Predictions")
    predictions = generate_predictions()
    predictions_df = pd.DataFrame(predictions, columns=[f"N{i+1}" for i in range(NUM_MAIN + NUM_SUPP)])
    st.dataframe(predictions_df)
    csv = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Predictions", csv, "oz_lotto_predictions.csv", "text/csv")
else:
    st.warning("⚠️ Upload a CSV file to proceed.")
