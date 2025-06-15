
# 🧠 Big Win Oz Lotto – Hybrid Predictor App

This Streamlit app predicts Oz Lotto numbers using a hybrid mathematical formula and optionally applies a machine learning model for enhanced prediction accuracy.

## 🔧 Features
- Upload your own historical draw data (.csv)
- Upload a trained ML model (.pkl, e.g., RandomForest or LightGBM)
- Adjust prediction formula weights in real time:
  - Alpha: Frequency weight
  - Beta: Hot zone score
  - Gamma: Cold number bonus
- Predicts 7 main numbers and 2 supplementary numbers
- Scores predictions with:
  - Hybrid scoring formula
  - Optional ML model scoring
- Exports top predictions as downloadable CSV

## 📁 Files
- `oz_lotto_hybrid_predictor.py`: Streamlit app
- `requirements.txt`: Required dependencies
- `README.md`: This file

## 🚀 Getting Started

```bash
pip install -r requirements.txt
streamlit run oz_lotto_hybrid_predictor.py
```

## 📤 Deploy on Streamlit Cloud
Upload the repo and use Streamlit Cloud to launch it online. Upload your own historical draws and model during runtime!

