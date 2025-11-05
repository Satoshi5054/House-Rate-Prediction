import streamlit as st
#import pandas as pd
#import pickle, json
#from src.features import predict_price_simple

st.title("🏠 Property Price Predictor")
"""
# Load artifacts
with open("models/feature_meta.json") as f:
    meta = json.load(f)
numerical_features = meta["numerical_features"]
categorical_features = meta["categorical_features"]

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
"""
# UI Inputs
st.subheader("Enter Property Details:")
input_data = {
    "State": [st.selectbox("State", ["Maharashtra"])],
    "City": [st.selectbox("City", ["Pune"])],
    "Locality": [st.text_input("Locality", "Locality_490")],
    "Property_Type": [st.selectbox("Property Type", ["Independent House", "Apartment"])],
    "BHK": [st.number_input("BHK", 1, 10, 3)],
    "Size_in_SqFt": [st.number_input("Size (SqFt)", 200, 10000, 2364)],
    "Price_per_SqFt": [st.number_input("Price per SqFt (Lakhs)", 0.01, 10.0, 0.08, step=0.01)],
    "Year_Built": [st.number_input("Year Built", 1900, 2025, 2008)],
    "Furnished_Status": [st.selectbox("Furnished Status", ["Unfurnished", "Furnished"])],
    "Floor_No": [st.number_input("Floor No", 0, 50, 2)],
    "Total_Floors": [st.number_input("Total Floors", 1, 100, 3)],
    "Age_of_Property": [st.number_input("Age of Property (Years)", 0, 100, 17)],
    "Nearby_Schools": [st.number_input("Nearby Schools", 0, 20, 8)],
    "Nearby_Hospitals": [st.number_input("Nearby Hospitals", 0, 20, 1)],
    "Public_Transport_Accessibility": [st.selectbox("Transport Accessibility", ["Low", "Medium", "High"])],
    "Parking_Space": [st.selectbox("Parking Space", ["Yes", "No"])],
    "Security": [st.selectbox("Security", ["Yes", "No"])],
    "Amenities": [st.text_input("Amenities", "Playground, Clubhouse")],
    "Facing": [st.selectbox("Facing", ["North", "East", "South", "West"])],
    "Owner_Type": [st.selectbox("Owner Type", ["Owner", "Builder", "Agent"])],
    "Availability_Status": [st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])]
}
#new_df = pd.DataFrame(input_data)
"""
# Prediction
if st.button("Predict Price"):
    price = model.predict(pd.get_dummies(new_df))[0]
    st.success(f"Predicted Price (Lakhs): {price:.2f}")
"""