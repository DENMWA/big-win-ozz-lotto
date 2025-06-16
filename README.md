# Big Win Ozz Lotto 🎯

This Streamlit app predicts Oz Lotto numbers using a hybrid system that combines statistical filters, machine learning, and entropy-based randomness. Users can upload new historical datasets and trained ML models to dynamically update predictions.

## Features

- 🎰 Hybrid formula combining hot/cold zones, frequency, entropy, and Mahalanobis distance
- 🤖 Upload and apply a trained ML model to score predictions
- 📊 Dynamic sliders for α, β, γ weight adjustment
- 📁 Upload CSV historical data directly from the app
- 🔮 Predicts 7 main and 2 supplementary numbers
- 📤 Export top predictions as CSV

## How to Run

### Locally:
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run Streamlit:
```bash
streamlit run streamlit_app.py
```

### On Streamlit Cloud:
1. Push to GitHub
2. Deploy through https://streamlit.io/cloud
3. Upload your historical `.csv` and optional `.pkl` model

---
**Good luck!** 🍀 May this be the path to your Big Win.
