import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your dataset
df = pd.read_csv('example1.csv')  # Update with the actual path to your dataset

# Selecting features and target
X = df[['Age', 'Potential']]
y = df['Release Clause']

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor model with parameters to reduce overfitting
model = RandomForestRegressor(
    n_estimators=100,               # Number of trees in the forest
    max_depth=10,                   # Maximum depth of the tree
    min_samples_split=10,           # Minimum number of samples required to split a node
    min_samples_leaf=4,             # Minimum number of samples required at a leaf node
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse = -cross_val_score(model, X_normalized, y, cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(cv_mse)
cv_r2 = cross_val_score(model, X_normalized, y, cv=kf, scoring='r2')

# Streamlit App
st.title('Football Player Release Clause Prediction')

st.sidebar.header('Input Data')
age = st.sidebar.slider('Select Age', min_value=15, max_value=40, value=25)
potential = st.sidebar.slider('Select Potential', min_value=60, max_value=100, value=70)

# Normalize the input data
input_data = scaler.transform([[age, potential]])
predicted_release_clause = model.predict(input_data)

st.write(f"The predicted release clause for a player with age {age} and potential {potential} is: €{int(predicted_release_clause[0]):,}")

st.header('Model Evaluation Metrics')
st.write(f"Train Mean Squared Error (MSE): {train_mse:.2f}")
st.write(f"Train Root Mean Squared Error (RMSE): {train_rmse:.2f}")
st.write(f"Train Mean Absolute Error (MAE): {train_mae:.2f}")
st.write(f"Train R2 Score: {train_r2:.2f}")
st.write(f"Test Mean Squared Error (MSE): {test_mse:.2f}")
st.write(f"Test Root Mean Squared Error (RMSE): {test_rmse:.2f}")
st.write(f"Test Mean Absolute Error (MAE): {test_mae:.2f}")
st.write(f"Test R2 Score: {test_r2:.2f}")
st.write(f"Cross-Validated Mean Squared Error (MSE): {np.mean(cv_mse):.2f}")
st.write(f"Cross-Validated Root Mean Squared Error (RMSE): {np.mean(cv_rmse):.2f}")
st.write(f"Cross-Validated R2 Score: {np.mean(cv_r2):.2f}")

# Feature Importance
st.header('Feature Importance')
importance = model.feature_importances_
features = ['Age', 'Potential']
fig, ax = plt.subplots()
sns.barplot(x=importance, y=features, ax=ax)
ax.set_title('Feature Importance')
st.pyplot(fig)

# Distribution of Predicted vs Actual Values
st.header('Distribution of Predicted vs Actual Values')
fig, ax = plt.subplots()
sns.histplot(y_test, color='blue', label='Actual', kde=True, ax=ax)
sns.histplot(y_test_pred, color='orange', label='Predicted', kde=True, ax=ax)
ax.legend()
ax.set_title('Distribution of Predicted vs Actual Values')
st.pyplot(fig)
