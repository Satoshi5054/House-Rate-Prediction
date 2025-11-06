import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- 1. LOAD ARTIFACTS ---
# These files are created by running the 'train_and_save.py' script locally.
# Load the pre-trained model
model = joblib.load('artifacts/model.pkl')
# Load the fitted scaler
scaler = joblib.load('artifacts/scaler.pkl')
# Load the artifacts (median values, column lists, etc.)
with open('artifacts/model_artifacts.json', 'r') as f:
    artifacts = json.load(f)

numerical_features = artifacts['numerical_features']
categorical_features = artifacts['categorical_features']
model_columns = artifacts['model_columns']
median_values = artifacts['median_values']
target_variable = artifacts['target_variable']


# --- 2. BUILD THE USER INTERFACE (UI) ---
st.set_page_config(page_title="Indian House Price Predictor", layout="wide")
st.title('🏠 Indian House Price Predictor')
st.write("Enter the details of the property to get an estimated price.")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Location & Property")
    State = st.text_input('State', 'Maharashtra')
    City = st.text_input('City', 'Pune')
    Locality = st.text_input('Locality', 'Locality_490')
    Property_Type = st.selectbox('Property Type', ['Independent House', 'Apartment', 'Villa', 'Builder Floor'])
    Facing = st.selectbox('Facing', ['North', 'South', 'East', 'West', 'North-East', 'North-West', 'South-East', 'South-West', 'Missing'])
    
    st.subheader("Property Status")
    Furnished_Status = st.radio("Furnished Status", ['Unfurnished', 'Semi-Furnished', 'Furnished'], index=0)
    Owner_Type = st.radio("Owner Type", ['Builder', 'Resale', 'New'], index=0)
    Availability_Status = st.radio("Availability Status", ['Under_Construction', 'Ready_to_Move', 'New_Launch'], index=0)
    Nearby_Schools = st.slider('Nearby Schools (Count)', 0, 20, 8)
    Nearby_Hospitals = st.slider('Nearby Hospitals (Count)', 0, 20, 1)

with col2:
    st.subheader("Size & Build")
    BHK = st.number_input('BHK', min_value=1, max_value=10, value=3)
    Size_in_SqFt = st.number_input('Size in SqFt', min_value=100, max_value=10000, value=2364)
    Price_per_SqFt = st.number_input('Price per SqFt (approx)', min_value=0.01, max_value=5.0, value=0.08, step=0.01, format="%.2f")
    Year_Built = st.number_input('Year Built', min_value=1950, max_value=2025, value=2008)
    Floor_No = st.number_input('Floor No', min_value=0, max_value=50, value=2)
    Total_Floors = st.number_input('Total Floors', min_value=0, max_value=50, value=3)
    Age_of_Property = st.number_input('Age of Property (Years)', min_value=0, max_value=100, value=17)

    st.subheader("Amenities & Accessibility")
    Public_Transport_Accessibility = st.radio("Public Transport", ['Low', 'Medium', 'High', 'Missing'], index=0)
    Parking_Space = st.radio("Parking Space", ['No', 'Yes', 'Missing'], index=0)
    Security = st.radio("Security", ['No', 'Yes', 'Missing'], index=0)
    


# --- 3. PREDICTION LOGIC ---
if st.button('Estimate Price', type="primary"):
    
    # --- A. Collect UI data into a dictionary ---
    new_data = {
        'State': [State],
        'City': [City],
        'Locality': [Locality],
        'Property_Type': [Property_Type],
        'BHK': [BHK],
        'Size_in_SqFt': [Size_in_SqFt],
        'Price_per_SqFt': [Price_per_SqFt],
        'Year_Built': [Year_Built],
        'Furnished_Status': [Furnished_Status],
        'Floor_No': [Floor_No],
        'Total_Floors': [Total_Floors],
        'Age_of_Property': [Age_of_Property],
        'Nearby_Schools': [Nearby_Schools],
        'Nearby_Hospitals': [Nearby_Hospitals],
        'Public_Transport_Accessibility': [Public_Transport_Accessibility],
        'Parking_Space': [Parking_Space],
        'Security': [Security],
        'Facing': [Facing],
        'Owner_Type': [Owner_Type],
        'Availability_Status': [Availability_Status]
        # 'Price_in_Lakhs' is excluded as it's the target
    }
    
    # Convert to DataFrame
    new_df = pd.DataFrame(new_data)

    # --- B. Apply feature engineering steps (matching the training data) ---
    
    # Handle missing values (using the saved median_values from training)
    for col in numerical_features:
        if new_df[col].isnull().sum() > 0:
            new_df[col] = new_df[col].fillna(median_values[col])

    for col in categorical_features:
        if new_df[col].isnull().sum() > 0:
            new_df[col] = new_df[col].fillna('Missing')

    # Scale numerical features using the LOADED scaler
    new_df_scaled_num = pd.DataFrame(scaler.transform(new_df[numerical_features]),
                                     columns=numerical_features,
                                     index=new_df.index)

    # One-Hot Encode categorical features
    new_df_encoded_cat = pd.get_dummies(new_df[categorical_features], drop_first=True, dummy_na=False)

    # Combine processed features
    X_new_processed = pd.concat([new_df_scaled_num, new_df_encoded_cat], axis=1)
    
    # --- C. Align columns ---
    # This is CRITICAL. Ensures the new data has the exact same columns as the training data.
    # It adds missing dummy columns (with 0) and removes extra columns (not in training).
    X_new_aligned = X_new_processed.reindex(columns=model_columns, fill_value=0)

    # --- D. Make prediction ---
    try:
        predicted_price = model.predict(X_new_aligned)
        
        # --- E. Display result ---
        st.subheader('Prediction Result')
        prediction_text = f"₹ {predicted_price[0]:,.2f} Lakhs"
        st.metric(label=f"Estimated {target_variable.replace('_', ' ')}", value=prediction_text)
        
        st.success("Prediction successful!")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check your inputs and try again.")
        st.write("Debug info (Aligned Columns):", X_new_aligned.columns)
        st.write("Debug info (Model Columns):", model_columns)