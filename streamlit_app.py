
# oz_lotto_hybrid_predictor.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial import distance
import joblib
import os

NUMBERS_RANGE = list(range(1, 48))
NUM_MAIN = 7
NUM_SUPP = 2
NUM_SETS = 100

st.sidebar.header("Weight Adjustment (Final Formula)")
alpha = st.sidebar.slider("Alpha – Frequency Weight", 0.0, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("Beta – Hot Zone Weight", 0.0, 2.0, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma – Cold Zone Weight", 0.0, 2.0, 1.0, 0.1)

st.title("🧠 Oz Lotto Hybrid Predictor with Supplementaries + ML Model")
st.markdown("---")

st.markdown("### 📂 Upload Files")
st.markdown("**Required:** Historical CSV with at least 7 main number columns")

uploaded_csv = st.file_uploader("Upload Historical Data (.csv)", type=["csv"], key="csv")
uploaded_model = st.file_uploader("Upload Trained ML Model (.pkl)", type=["pkl"], key="model")

csv_path = "historical_data.csv"
model_path = "oz_lotto_model.pkl"

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        if df.shape[1] < 7:
            st.error("❌ CSV must contain at least 7 columns of numbers.")
            st.stop()
        df.to_csv(csv_path, index=False)
        st.success("✅ Historical data uploaded and saved.")
    except Exception as e:
        st.error(f"❌ Error processing CSV: {e}")
        st.stop()
elif os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.info("ℹ️ Using previously uploaded historical data.")
else:
    st.warning("⚠️ Please upload historical data to proceed.")
    st.stop()

if uploaded_model is not None:
    with open(model_path, "wb") as f:
        f.write(uploaded_model.read())
    st.success("✅ ML model uploaded and saved.")

st.dataframe(df.head())

all_numbers = df.iloc[:, 1:8].values.flatten()
historical_freq = pd.Series(all_numbers).value_counts().sort_index()
for num in NUMBERS_RANGE:
    if num not in historical_freq:
        historical_freq[num] = 0
historical_freq = historical_freq.sort_index()
st.bar_chart(historical_freq)

def build_ml_features(draw):
    draw = sorted(draw)
    return {
        'mean': np.mean(draw),
        'std': np.std(draw),
        'entropy': entropy([historical_freq.get(n, 0)/historical_freq.sum() for n in draw]),
        'gap_sum': sum(np.diff(draw)),
        'hot_count': sum([1 if historical_freq.get(n, 0) >= np.percentile(historical_freq, 75) else 0 for n in draw]),
        'cold_count': sum([1 if historical_freq.get(n, 0) <= np.percentile(historical_freq, 25) else 0 for n in draw])
    }

ml_features_df = pd.DataFrame([build_ml_features(row[1:8]) for row in df.itertuples()], dtype=np.float32)
st.write("📊 Sample ML Features", ml_features_df.head())

try:
    model = joblib.load(model_path)
    st.success("🧠 Machine Learning Model Loaded")
except:
    model = None
    st.warning("⚠️ ML model not found or failed to load.")

def hot_zone_score(freqs):
    zone_thresh = np.percentile(freqs, 75)
    return pd.Series((freqs >= zone_thresh).astype(int), index=freqs.index)

def cold_zone_score(freqs):
    zone_thresh = np.percentile(freqs, 25)
    return pd.Series((freqs <= zone_thresh).astype(int), index=freqs.index)

def sequential_penalty(numbers):
    numbers = sorted(numbers)
    return sum(1 for i in range(len(numbers) - 1) if numbers[i+1] - numbers[i] == 1)

def entropy_score(numbers):
    probs = np.array([historical_freq[n]/historical_freq.sum() for n in numbers])
    return entropy(probs, base=2)

def mahalanobis_distance(numbers, historical_matrix):
    if len(historical_matrix) < 2:
        return 0
    mu = np.mean(historical_matrix, axis=0)
    cov = np.cov(historical_matrix, rowvar=False)
    try:
        return distance.mahalanobis(numbers, mu, np.linalg.inv(cov))
    except:
        return 0

def generate_mode_c_predictions():
    predictions = []
    hot_scores = hot_zone_score(historical_freq)
    cold_scores = cold_zone_score(historical_freq)
    for _ in range(NUM_SETS):
        scores = historical_freq.copy()
        scores += np.random.randn(47) * 0.5
        scores += hot_scores * 1.5
        scores -= cold_scores * 0.5
        probs = scores / scores.sum()
        mains = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
        remaining = list(set(NUMBERS_RANGE) - set(mains))
        supps = np.random.choice(remaining, size=NUM_SUPP, replace=False)
        predictions.append(sorted(mains) + sorted(supps))
    return predictions

def evaluate_final_formula(predictions, historical_matrix):
    hot_scores = hot_zone_score(historical_freq)
    cold_scores = cold_zone_score(historical_freq)
    scored = []
    for entry in predictions:
        mains = entry[:NUM_MAIN]
        F = np.mean([historical_freq[n]/historical_freq.sum() for n in mains])
        H = np.mean([hot_scores.get(n, 0) for n in mains])
        C = np.mean([cold_scores.get(n, 0) for n in mains])
        S = sequential_penalty(mains)
        E = entropy_score(mains)
        M = mahalanobis_distance(mains, historical_matrix)
        score = (alpha*F + beta*H + gamma*C) - S + E + M
        scored.append((entry, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

st.subheader("🧪 Prediction Simulation")
mode_c_preds = generate_mode_c_predictions()
historical_matrix = [sorted(np.random.choice(NUMBERS_RANGE, NUM_MAIN, replace=False)) for _ in range(50)]
evaluated = evaluate_final_formula(mode_c_preds, historical_matrix)

st.markdown("### 🔝 Top 10 Predicted Sets (7 Main + 2 Supp)")
top_df = pd.DataFrame([x[0] for x in evaluated[:10]], columns=[f"N{i+1}" for i in range(NUM_MAIN + NUM_SUPP)])
top_df["Score"] = [round(x[1], 3) for x in evaluated[:10]]

if model:
    st.markdown("### 🤖 ML Model Scoring")
    def build_features_for_model(entry):
        return build_ml_features(entry[:NUM_MAIN])
    features = pd.DataFrame([build_features_for_model(x[0]) for x in evaluated[:10]])
    ml_scores = model.predict(features)
    top_df["ML Score"] = np.round(ml_scores, 3)

st.dataframe(top_df)
csv = top_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇ Download Top Predictions", csv, "oz_lotto_predictions.csv", "text/csv")
