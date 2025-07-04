
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🎯 OzWinner Lotto Predictor with Live ML Training")
st.markdown("Upload your historical Oz Lotto data and train a model directly inside this app.")

uploaded_file = st.file_uploader("Upload Historical Data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")
    st.dataframe(df.head())

    # Basic feature engineering (example: even/odd counts)
    st.markdown("### ✅ Extracting Features...")
    features = df[[f'Winning Number {i}' for i in range(1, 8)]].values
    labels = np.array([1 if 1 in row else 0 for row in features])

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    if st.button("🚀 Train Model"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model trained successfully with accuracy: {accuracy:.2f}")

        st.markdown("### 🎯 Predictions on New Random Draws")
        for _ in range(5):
            simulated = np.random.choice(range(1, 48), size=7, replace=False)
            simulated = simulated.reshape(1, -1)
            pred = model.predict(simulated)
            st.write(f"Numbers: {simulated.flatten()} | Predicted Win-Likelihood: {pred[0]}")

else:
    st.warning("📥 Please upload a valid CSV file to begin.")
