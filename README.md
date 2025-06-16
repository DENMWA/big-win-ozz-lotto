# Big Win Ozz Lotto - ML Ready

This is a Streamlit-based predictive engine for Oz Lotto using a hybrid statistical + machine learning model.

## Features
- Upload historical Oz Lotto data (CSV)
- Auto-train LightGBM model from historical data
- Generate top 10 number predictions (7 main + 2 supplementary)
- Tune weights: frequency, hot zone, cold zone
- Optional ML scoring (if model is trained)
- Download top predictions as CSV

## Usage
1. Run `streamlit run streamlit_app.py`
2. Upload historical CSV
3. Adjust weights
4. View predictions and optionally train/score with ML

## Requirements
Install dependencies via:
```bash
pip install -r requirements.txt
```
