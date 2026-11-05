
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥"
)

st.title("🏥 Personalized Healthcare Recommendation")
st.write("Select your symptoms to see an educational prediction.")

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load saved files
try:
    with open(MODEL_DIR / "healthcare_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODEL_DIR / "symptom_encoder.pkl", "rb") as f:
        mlb = pickle.load(f)

    disease_info = pd.read_csv(MODEL_DIR / "disease_info.csv")

except Exception as e:
    st.error("Unable to load the saved model or data files.")
    st.code(str(e))
    st.stop()

# Symptom selection
symptoms = sorted(mlb.classes_.tolist())

selected_symptoms = st.multiselect(
    "Select your symptoms:",
    symptoms
)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_data = mlb.transform([selected_symptoms])
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.success(f"Possible Disease: {prediction}")

        result = disease_info[
            disease_info["Disease"].str.strip().str.lower()
            == prediction.strip().lower()
        ]

        if not result.empty:
            st.subheader("Disease Description")
            st.write(result.iloc[0]["Description"])

        st.info(
            "Educational prediction only. This is not a medical diagnosis "
            "or treatment recommendation. Please consult a qualified "
            "healthcare professional."
        )

st.divider()
st.caption("Developed as a Data Science and Machine Learning internship project.")
