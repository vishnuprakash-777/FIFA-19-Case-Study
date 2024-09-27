import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (assuming df is your DataFrame)
# Replace this with your actual dataset loading code
# For demonstration purposes, we'll create a sample DataFrame
df=pd.read_csv('example1.csv')

# Encode Preferred Foot
encoder = LabelEncoder()
df['Preferred_Foot_Encoded'] = encoder.fit_transform(df['Preferred Foot'])

# Define features and target
features = ['Attacking_Ability', 'Defensive_Ability', 'Physical_Ability', 
            'Goalkeeping_Ability', 'Playmaking_Ability', 'Speed', 'Technical_Skills', 'Preferred_Foot_Encoded']
target = 'Position'

X = df[features]
y = df[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Function to display classification report and confusion matrix
def display_metrics(y_test, y_pred):
    st.header("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
   

    # Visualize confusion matrix as a heatmap
   # Visualize confusion matrix as a heatmap
    st.subheader("Confusion Matrix Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


# Streamlit App
st.title("Football Player Position Prediction")

# Input form for new player attributes
st.sidebar.header("Enter New Player Attributes:")
attacking_ability = st.sidebar.slider("Attacking Ability", min_value=0, max_value=100, value=75)
defensive_ability = st.sidebar.slider("Defensive Ability", min_value=0, max_value=100, value=75)
physical_ability = st.sidebar.slider("Physical Ability", min_value=0, max_value=100, value=75)
goalkeeping_ability = st.sidebar.slider("Goalkeeping Ability", min_value=0, max_value=100, value=75)
playmaking_ability = st.sidebar.slider("Playmaking Ability", min_value=0, max_value=100, value=75)
speed = st.sidebar.slider("Speed", min_value=0, max_value=100, value=75)
technical_skills = st.sidebar.slider("Technical Skills", min_value=0, max_value=100, value=75)
preferred_foot = st.sidebar.radio("Preferred Foot", ['Right', 'Left'])

# Predicting the position for the new player
new_player = pd.DataFrame({
    'Attacking_Ability': [attacking_ability],
    'Defensive_Ability': [defensive_ability],
    'Physical_Ability': [physical_ability],
    'Goalkeeping_Ability': [goalkeeping_ability],
    'Playmaking_Ability': [playmaking_ability],
    'Speed': [speed],
    'Technical_Skills': [technical_skills],
    'Preferred_Foot': [preferred_foot]
})

# Encode Preferred Foot and predict position
new_player['Preferred_Foot_Encoded'] = encoder.transform(new_player['Preferred_Foot'])
predicted_position = clf.predict(new_player[features])

# Display prediction
st.header("Predicted Position for the New Player:")
st.write(predicted_position[0])

# Display model evaluation metrics
st.sidebar.header("Model Evaluation Metrics:")
display_metrics(y_test, clf.predict(X_test))
