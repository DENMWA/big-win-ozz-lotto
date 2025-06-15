# Big Win Ozz Lotto 🎯

A Streamlit-based app that predicts 7 main numbers and 2 supplementary numbers for Oz Lotto using statistical modeling and entropy scoring.

## Features
- Predicts 7 main numbers + 2 supps from 1–47
- Tunable weights for frequency, hot zone, cold zone factors
- Entropy and Mahalanobis scoring
- Downloadable top-10 ranked sets

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud
1. Push to GitHub
2. Connect on [streamlit.io/cloud](https://streamlit.io/cloud)
3. Set `streamlit_app.py` as main entry point
