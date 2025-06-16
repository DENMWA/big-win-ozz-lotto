
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import entropy
from io import StringIO

st.set_page_config(page_title="Big Win Oz Lotto", layout="centered")

st.title("🎯 Big Win Oz Lotto – Hybrid Predictor with ML")
st.markdown("Upload historical data and let the system generate smart predictions using a hybrid model including machine learning (if available).")

# Sidebar sliders
alpha = st.sidebar.slider("Alpha – Frequency Weight", 0.0, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("Beta – Hot Zone Weight", 0.0, 2.0, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma – Cold Zone Weight", 0.0, 2.0, 1.0, 0.1)

# Constants
NUMBERS_RANGE = list(range(1, 48))
NUM_MAIN = 7
NUM_SUPP = 2
NUM_SETS = 50

# Upload section
st.subheader("📂 Upload Historical Draw Data")
csv_file = st.file_uploader("Upload Oz Lotto CSV", type=["csv"])
model_file = st.file_uploader("Upload ML Model (.pkl)", type=["pkl"])

if csv_file:
    df = pd.read_csv(csv_file)
    if df.shape[1] < 7:
        st.error("CSV must contain at least 7 main number columns.")
        st.stop()
    all_nums = df.iloc[:, :7].values.flatten()
    freq_series = pd.Series(all_nums).value_counts().sort_index()
    for num in NUMBERS_RANGE:
        if num not in freq_series:
            freq_series[num] = 0
    freq_series = freq_series.sort_index()
    st.success("✅ Historical data processed.")
    st.bar_chart(freq_series)

    # Score builders
    def hot_zone(freqs):
        return (freqs >= np.percentile(freqs, 75)).astype(int)

    def cold_zone(freqs):
        return (freqs <= np.percentile(freqs, 25)).astype(int)

    hot_scores = hot_zone(freq_series)
    cold_scores = cold_zone(freq_series)

    def generate_predictions():
        predictions = []
        for _ in range(NUM_SETS):
            scores = freq_series.copy()
            scores += np.random.randn(47)
            scores += hot_scores * 1.5
            scores -= cold_scores * 0.5
            probs = scores / scores.sum()
            mains = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
            remaining = list(set(NUMBERS_RANGE) - set(mains))
            supps = np.random.choice(remaining, size=NUM_SUPP, replace=False)
            predictions.append(sorted(mains) + sorted(supps))
        return predictions

    predictions = generate_predictions()

    st.subheader("🔮 Top Predictions")
    pred_df = pd.DataFrame(predictions, columns=[f"N{i+1}" for i in range(NUM_MAIN + NUM_SUPP)])

    if model_file:
        try:
            model = joblib.load(model_file)
            st.success("🧠 ML model loaded.")
            def extract_features(entry):
                entry = entry[:NUM_MAIN]
                hot = sum([1 if freq_series.get(n, 0) >= np.percentile(freq_series, 75) else 0 for n in entry])
                cold = sum([1 if freq_series.get(n, 0) <= np.percentile(freq_series, 25) else 0 for n in entry])
                return {
                    'mean': np.mean(entry),
                    'std': np.std(entry),
                    'entropy': entropy([freq_series.get(n, 0)/freq_series.sum() for n in entry]),
                    'hot_count': hot,
                    'cold_count': cold
                }

            features = pd.DataFrame([extract_features(p) for p in predictions])
            pred_df["ML Score"] = model.predict(features)
        except Exception as e:
            st.warning(f"⚠️ ML model could not be used: {e}")

    st.dataframe(pred_df.head(10))
    st.download_button("⬇ Download All Predictions", pred_df.to_csv(index=False), "oz_predictions.csv", "text/csv")
else:
    st.info("📌 Please upload historical Oz Lotto data to continue.")
