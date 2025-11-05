import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  # Used to save the model and scaler
import json    # Used to save column names and medians

warnings.filterwarnings('ignore')

print("--- 1. Loading Data ---")
# --- PUT YOUR FILE NAME HERE ---
file_path = "/content/drive/MyDrive/House_Rate_Prediction/india_housing_prices.csv"
# Or if it's in the same folder:
# file_path = "india_housing_prices.csv" 

df = pd.read_csv(file_path)

# -----------------------------------------------------------------
# >>>> USER CONFIGURATION <<<<
# -----------------------------------------------------------------
target_variable = 'Price_in_Lakhs'
categorical_features = ['State', 'City', 'Locality', 'Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status']
numerical_features = ['BHK', 'Size_in_SqFt', 'Price_per_SqFt', 'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals']
# -----------------------------------------------------------------


print("\n--- 2. Pre-processing ---")

# --- A. Handle Missing Values & Save Medians ---
median_values = {} # Dictionary to store medians
for col in numerical_features:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    median_values[col] = median_val # Save the median for later
    print(f"Filled NaNs in '{col}' with median: {median_val}")

for col in categorical_features:
    df[col] = df[col].fillna('Missing')

# --- B. Feature Engineering ---

# 1. Scaling Numerical Features
print("Fitting StandardScaler and saving it.")
scaler = StandardScaler()
df_scaled_num = pd.DataFrame(scaler.fit_transform(df[numerical_features]),
                             columns=numerical_features,
                             index=df.index)
# Save the fitted scaler
joblib.dump(scaler, '/content/models/scaler.pkl')

# 2. One-Hot Encoding Categorical Features
df_encoded_cat = pd.get_dummies(df[categorical_features], drop_first=True, dummy_na=False)

# 3. Combine Processed Features
X = pd.concat([df_scaled_num, df_encoded_cat], axis=1)
y = df[target_variable]

# Save the final model columns
model_columns = X.columns.tolist()

 ## 3. Exploratory Data Analysis (EDA) & Visualization
# This section remains the same, as it analyzes the *original* data
print("\n--- 3. Visualizing Original Data ---")

# Plot 1: Distribution of the Target Variable
if target_variable in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_variable], kde=True, bins=50)
    plt.title(f'Distribution of {target_variable}')
    plt.xlabel(target_variable)
    plt.ylabel('Frequency')
    plt.show()

# Plot 2: Numerical Feature vs. Target (Example: first numerical feature)
if numerical_features and target_variable in df.columns:
    num_feat_to_plot = numerical_features[0]
    plt.figure(figsize=(10, 6))
    # Plotting original (unscaled) data for better interpretation
    sns.scatterplot(x=df[num_feat_to_plot], y=df[target_variable], alpha=0.6)
    plt.title(f'{target_variable} vs. {num_feat_to_plot} (Original)')
    plt.xlabel(f'{num_feat_to_plot} (Original)')
    plt.ylabel(target_variable)
    plt.show()

# Plot 3: Categorical Feature vs. Target (Example: first categorical feature)
if categorical_features and target_variable in df.columns:
    cat_feat_to_plot = categorical_features[0]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[cat_feat_to_plot], y=df[target_variable])
    plt.title(f'{target_variable} by {cat_feat_to_plot}')
    plt.xlabel(cat_feat_to_plot)
    plt.ylabel(target_variable)
    plt.xticks(rotation=45)
    plt.show()

# Plot 4: Correlation Heatmap for numerical features
if numerical_features and target_variable in df.columns:
    plt.figure(figsize=(10, 6))
    corr_cols = numerical_features + [target_variable]
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap (Numerical Features + Target)')
    plt.show()

print("\n--- 4. Building and Saving Model ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose = 2
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Model training complete.")

## 5. Model Evaluation (3 Accuracy Checkers)
print("\n--- 5. Evaluating Model ---")
# This section is model-agnostic and remains the same

# Make predictions on the test set
y_pred = model.predict(X_test)
# 1. R-squared (R²) - Coefficient of Determination
r2 = r2_score(y_test, y_pred)
print(f"Accuracy Checker 1: R-squared (R²) = {r2:.4f}")

# 2. Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Accuracy Checker 2: Mean Absolute Error (MAE) = {mae:,.2f} (in units of '{target_variable}')")

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Accuracy Checker 3: Root Mean Squared Error (RMSE) = {rmse:,.2f} (in units of '{target_variable}')")


# Save the trained model
joblib.dump(model, '/content/models/model.pkl')
print("Saved model to model.pkl")

## 6. Graphical Representation of Model Results
print("\n--- 6. Visualizing Model Results ---")

# Plot 1: Feature Importance
try:
  importances = model.feature_importances_
  feature_names = X.columns

  # Create a DataFrame for easy plotting
  feature_importance_df = pd.DataFrame({
          'Feature': feature_names,
          'Importance': importances
  }).sort_values(by='Importance', ascending=False)

  plt.figure(figsize=(12, 8))
  sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20)) # Plot top 20
  plt.title('Random Forest Feature Importance (Top 20)')
  plt.show()

except Exception as e:
  print(f"Could not plot feature importance: {e}")

# Plot 2: Predictions vs. Actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
# Add a 45-degree line for reference (perfect prediction)
line_max = max(y_test.max(), y_pred.max())
line_min = min(y_test.min(), y_pred.min())
plt.plot([line_min, line_max], [line_min, line_max], '--', color='red', lw=2)
plt.title('Actual vs. Predicted Values')
plt.xlabel(f'Actual {target_variable}')
plt.ylabel(f'Predicted {target_variable}')
plt.show()

print("\nScript halted. Please fix the file path or check the CSV file.")

# --- 5. Save Artifacts ---
# Save the medians, column lists, and target variable name to a JSON file
artifacts = {
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "model_columns": model_columns,
    "median_values": median_values,
    "target_variable": target_variable
}

with open('/content/models/model_artifacts.json', 'w') as f:
    json.dump(artifacts, f)

print("Saved model artifacts (columns, medians) to model_artifacts.json")
print("\n--- Training and saving process complete. ---")
print("You now have model.pkl, scaler.pkl, and model_artifacts.json")