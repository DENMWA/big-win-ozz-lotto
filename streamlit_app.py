# oz_lotto_hybrid_predictor.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial import distance

# ----- Constants and Configs ----- #
NUMBERS_RANGE = list(range(1, 48))  # Oz Lotto: 1 to 47
NUM_MAIN = 7
NUM_SUPP = 2
NUM_SETS = 100

# ----- User Input Weights ----- #
st.sidebar.header("Weight Adjustment (Final Formula)")
alpha = st.sidebar.slider("Alpha – Frequency Weight", 0.0, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("Beta – Hot Zone Weight", 0.0, 2.0, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma – Cold Zone Weight", 0.0, 2.0, 1.0, 0.1)

st.title("🧠 Oz Lotto Hybrid Predictor with Supplementaries")
st.markdown("---")

# ----- Simulated Historical Data (Placeholder) ----- #
st.subheader("Simulated Historical Frequencies")
np.random.seed(42)
historical_freq = pd.Series(np.random.randint(5, 50, size=47), index=NUMBERS_RANGE)
st.bar_chart(historical_freq)

# ----- Formula Components ----- #
def fourier_score(freqs):
    ft = np.fft.fft(freqs)
    return np.abs(ft[:len(freqs)//2])

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

# ----- Prediction Generators ----- #
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
        H = np.mean([hot_scores[n] for n in mains])
        C = np.mean([cold_scores[n] for n in mains])
        S = sequential_penalty(mains)
        E = entropy_score(mains)
        M = mahalanobis_distance(mains, historical_matrix)
        score = (alpha*F + beta*H + gamma*C) - S + E + M
        scored.append((entry, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ----- Simulate and Display ----- #
st.subheader("🧪 Prediction Simulation")
mode_c_preds = generate_mode_c_predictions()
historical_matrix = [sorted(np.random.choice(NUMBERS_RANGE, NUM_MAIN, replace=False)) for _ in range(50)]
evaluated = evaluate_final_formula(mode_c_preds, historical_matrix)

st.markdown("### 🔝 Top 10 Predicted Sets (7 Main + 2 Supp)")
top_df = pd.DataFrame([x[0] for x in evaluated[:10]], columns=[f"N{i+1}" for i in range(NUM_MAIN + NUM_SUPP)])
top_df["Score"] = [round(x[1], 3) for x in evaluated[:10]]
st.dataframe(top_df)

csv = top_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇ Download Top Predictions", csv, "oz_lotto_predictions.csv", "text/csv")
