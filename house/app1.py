import streamlit as st
import pickle
import numpy as np
import json

st.set_page_config(page_title="Bangalore Home Price Estimator", layout="wide")

# Load model
with open("house/banglore_home_prices_model1.pickle", "rb") as f:
    model = pickle.load(f)

with open("house/model_mape.pickle", "rb") as f:
    mape = pickle.load(f)

with open("house/model_r2.pickle", "rb") as f:
    r2 = pickle.load(f)

# Load column names from JSON
with open("house/columns1.json", "r") as f:
    data_columns = json.load(f)['data_columns']

# Extract feature names
column_names = data_columns
locations = column_names[3:]  # First 3: sqft, bath, bhk

# UI
st.title("🏡 Bangalore Home Price Estimator")

sqft = st.number_input("Area (Square Feet)", min_value=100, max_value=10000, step=50, value=1000)
bath = st.radio("BHK", [1, 2, 3, 4, 5], horizontal=True)
bhk = st.radio("Bath", [1, 2, 3, 4, 5], horizontal=True)
location = st.selectbox("Location", sorted(locations))

if st.button("Estimate Price"):
    try:
        # Input vector (zeroes)
        x = np.zeros(len(column_names))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # One-hot encode location
        if location.lower() in column_names:
            loc_index = column_names.index(location.lower())
            x[loc_index] = 1

        # Predict
        price = model.predict([x])[0]
        lower = price * (1 - mape)
        upper = price * (1 + mape)

        st.success(f"💰 Estimated Price: ₹ {round(price * 1e5):,} with {round(r2 * 100, 2)}% accuracy")
        st.info(f"📊 Price Range (±{int(mape * 100)}%): ₹ {round(lower * 1e5):,} to ₹ {round(upper * 1e5):,}")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
