
import streamlit as st
import pandas as pd
import pickle

# Load saved files
with open("models/healthcare_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/symptom_encoder.pkl", "rb") as f:
    mlb = pickle.load(f)

disease_info = pd.read_csv("models/disease_info.csv")

# Page settings
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥"
)

st.title("🏥 Personalized Healthcare Recommendation")
st.write("Select your symptoms to see an educational prediction.")

# Symptom selection
symptoms = sorted(mlb.classes_.tolist())

selected_symptoms = st.multiselect(
    "Select your symptoms:",
    symptoms
)

# Prediction button
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")

    else:
        input_data = mlb.transform([selected_symptoms])
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.success(f"Possible Disease: {prediction}")

        # Disease description
        result = disease_info[
            disease_info["Disease"].str.strip().str.lower()
            == prediction.strip().lower()
        ]

        if not result.empty:
            st.subheader("Disease Description")
            st.write(result.iloc[0]["Description"])

        st.info(
            "This is an educational machine learning project. "
            "It does not provide medical diagnosis or treatment. "
            "Please consult a qualified healthcare professional."
        )

st.divider()
st.caption("Developed as a Data Science and Machine Learning internship project.")
