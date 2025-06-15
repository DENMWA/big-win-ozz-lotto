
# Big Win Ozz Lotto 🎉

A **Streamlit-powered hybrid prediction engine** for Australia's Oz Lotto. Combines statistical modeling, machine learning, and user-adjustable weights to generate high-quality predictions.

---

## 🔍 Features

### Hybrid Predictive Formula
- Weighted scoring using:
  - α (Alpha): Frequency weight
  - β (Beta): Hot zone occurrence
  - γ (Gamma): Cold zone rarity
- Supplementary number prediction support

### Machine Learning Model Integration
- Upload your trained `.pkl` model
- Live scoring of predictions using real-time feature engineering (entropy, std dev, co-draw probability, etc.)

### Upload Support
- Upload historical data as `.csv`
- Use custom number columns (no fixed format)

### Dynamic Simulation
- Generates 100 sets of 7 main + 2 supplementary numbers
- Evaluates with hybrid scoring + ML scoring (if model provided)

### Output
- Tabular predictions with score breakdown
- Downloadable results as `.csv`
- Bar chart of historical number frequencies

---

## ⚡ Requirements
Install with:
```bash
pip install -r requirements.txt
```

---

## 📂 Files
- `streamlit_app.py`: Main app
- `requirements.txt`: Python dependencies
- `README.md`: Project overview
- `oz_lotto_model.pkl`: Optional machine learning model (upload through UI)
- `historical_data.csv`: Historical Oz Lotto draws (upload through UI)

---

## ⚙️ How to Run
```bash
streamlit run streamlit_app.py
```

---

## 🎯 Goal
Help you build statistically and ML-informed predictions for Oz Lotto to maximize intelligent ticketing decisions.

> **Note**: This project is experimental and for educational purposes. Lotto outcomes are inherently random and no prediction system guarantees winnings.

---

## ✨ Upcoming Features
- Reinforcement learning from past draw results
- Genetic algorithms to evolve predictions
- Time-series modeling (e.g. LSTM)

---

## 🚀 Powered By:
Streamlit, NumPy, pandas, scikit-learn, joblib, SciPy
