import streamlit as st
import numpy as np
import pickle

# 1. Load the saved scaler and model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_diabetes(input_data):
    """
    Given a list/array of input features, preprocess and predict diabetes status.
    """
    input_array = np.asarray(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    # you can also get probability if model supports it
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

def main():
    st.title("Diabetes Prediction Web App")
    st.write("Enter patient details to predict whether they are diabetic or not.")

    # Input widgets for each feature (update names/ranges as per your dataset)
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose     = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    bp          = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thick  = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin     = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi         = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf         = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
    age         = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

    if st.button("Predict"):
        input_features = [pregnancies, glucose, bp, skin_thick, insulin, bmi, dpf, age]
        result = predict_diabetes(input_features)
        st.success(f"Prediction: **{result}**")

if __name__ == '__main__':
    main()
