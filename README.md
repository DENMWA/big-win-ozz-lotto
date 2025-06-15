# 🧠 Big Win Ozz Lotto – Hybrid Prediction Engine

This is a powerful Streamlit-based app for predicting Oz Lotto outcomes using:
- Statistical filters (entropy, hot/cold zones, Fourier)
- Hybrid scoring formula (with adjustable weights α, β, γ)
- Optional Machine Learning integration (LightGBM / Random Forest)
- Support for supplementary numbers
- Upload historical data for full feature engineering and real-time prediction

## 🚀 Features

- Upload and persist historical draw data (`.csv`)
- Generate 7 + 2 number sets using smart weighted formula
- Score predictions using entropy, gap spacing, and Mahalanobis distance
- Load and apply a trained machine learning model (`oz_lotto_model.pkl`)
- Visualize and download the top 10 predictions

## 🧪 ML Model Support

To train your own `.pkl` model:
```
python train_oz_lotto_model.py
```

## 📦 Requirements

Install using:
```
pip install -r requirements.txt
```

## ▶️ Run Locally

```
streamlit run oz_lotto_hybrid_predictor.py
```

## 🌐 Deploy on Streamlit Cloud

Push this to GitHub and deploy directly from:
[https://streamlit.io/cloud](https://streamlit.io/cloud)

---
Crafted with 🎯 by ChatGPT and [Your Name]