import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Page config
st.set_page_config(page_title="Iris Classifier", page_icon="💮")

# Load trained model
@st.cache_resource
def load_model():
    model_path = "iris_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
        st.error(f"Model file '{model_path}' not found")
        return None

model = load_model()

# Title
st.title("🌸 Iris Species Predictor")
st.write("Enter flower measurements to predict the Iris species.")

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create input array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
if st.button("Predict 🌼"):
    if model is not None:
        prediction = model.predict(input_data)
        species_map = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }
        st.success(f"🌺 Predicted Iris Species: **{species_map[prediction[0]]}**")
    else:
        st.warning("Model not loaded properly.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit 💻")
