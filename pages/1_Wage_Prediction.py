import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with your actual data loading method)
# Assume df is your DataFrame with 'Age', 'Overall', and 'Wage' columns
df = pd.read_csv('example1.csv')

# Selecting features and target
X = df[['Age', 'Overall']]
y = df['Wage']

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI components
st.title('Football Player Wage Prediction')
st.write('## Predict Wage based on Age and Overall')

# Sidebar inputs for Age and Overall
age = st.slider('Select Age', 15, 40, 25)
overall = st.slider('Select Overall', 40, 100, 75)

# Normalize the input values
input_data = [[age, overall]]
input_data_normalized = scaler.transform(input_data)

# Make prediction
predicted_wage = model.predict(input_data_normalized)

# Display prediction
st.subheader('Prediction:')
st.write(f"Predicted Wage: ${predicted_wage[0]:,.2f}")

# Display evaluation metrics
st.write(f"Mean Squared Error (MSE): {mse:,.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
st.write(f"Mean Absolute Error (MAE): {mae:,.2f}")
st.write(f"R2-score: {r2:.2f}")

# Create a DataFrame for actual vs predicted values for Wage
results_df_wage = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the actual vs predicted values for Wage as a line chart
st.write('## Accuracy of Predicted Wage')
st.line_chart(results_df_wage)

# Feature Importance
st.write('## Feature Importance')
feature_importances = model.feature_importances_
features = ['Age', 'Overall']
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Display the plot in Streamlit
st.pyplot(plt)
