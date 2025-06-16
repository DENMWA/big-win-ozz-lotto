
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import entropy
from scipy.spatial import distance

st.title("🎯 Big Win Ozz Lotto Predictor")
st.markdown("Upload your historical Oz Lotto data and simulate winning number predictions.")

# Config
NUMBERS_RANGE = list(range(1, 48))
NUM_MAIN = 7
NUM_SUPP = 2
NUM_SETS = 100

# Sidebar controls
st.sidebar.header("Weight Adjustment")
alpha = st.sidebar.slider("Alpha – Frequency Weight", 0.0, 2.0, 1.0)
beta = st.sidebar.slider("Beta – Hot Zone Weight", 0.0, 2.0, 1.0)
gamma = st.sidebar.slider("Gamma – Cold Zone Weight", 0.0, 2.0, 1.0)

# Upload historical data
uploaded_file = st.file_uploader("Upload historical draw CSV (must include 7+ columns for main numbers)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 7:
        st.error("The file must contain at least 7 columns for main numbers.")
        st.stop()

    # Flatten all historical numbers to build frequency data
    main_numbers = df.iloc[:, :7].values.flatten()
    freq_series = pd.Series(main_numbers).value_counts().sort_index()
    for n in NUMBERS_RANGE:
        if n not in freq_series:
            freq_series[n] = 0
    freq_series = freq_series.sort_index()
    st.bar_chart(freq_series)

    # Prediction logic
    def generate_predictions():
        hot_thresh = np.percentile(freq_series, 75)
        cold_thresh = np.percentile(freq_series, 25)
        hot_scores = (freq_series >= hot_thresh).astype(int)
        cold_scores = (freq_series <= cold_thresh).astype(int)

        predictions = []
        for _ in range(NUM_SETS):
            scores = freq_series.copy()
            scores += np.random.randn(len(scores)) * 0.5
            scores += beta * hot_scores
            scores -= gamma * cold_scores
            probs = scores / scores.sum()
            mains = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
            remaining = list(set(NUMBERS_RANGE) - set(mains))
            supps = np.random.choice(remaining, size=NUM_SUPP, replace=False)
            predictions.append(sorted(mains) + sorted(supps))
        return predictions

    # Generate and display
    st.subheader("🔮 Top Predictions")
    predictions = generate_predictions()
    top_df = pd.DataFrame(predictions, columns=[f"N{i+1}" for i in range(NUM_MAIN + NUM_SUPP)])
    st.dataframe(top_df)

    # Download
    csv = top_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Predictions", csv, "oz_lotto_predictions.csv", "text/csv")
else:
    st.warning("Please upload historical data to continue.")
