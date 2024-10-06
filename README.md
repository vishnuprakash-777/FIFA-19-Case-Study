# FIFA 19 Player Insights

**Author:** Vishnu Prakash J  
**Project:** Predicting Wages, Release Clauses, and Player Positions in FIFA 19

## Overview

This project leverages machine learning techniques to analyze and predict key player attributes from the FIFA 19 Complete Player dataset. By applying regression and classification models, the project predicts three critical metrics in football: player wages, release clauses, and positions. The analysis offers insights that could aid in financial planning, scouting, and team composition.

## Dataset

- **Source:** [Kaggle FIFA 19 Complete Player Dataset]([https://www.kaggle.com/karangadiya/fifa19](https://drive.google.com/file/d/1xEb00xhqoa99g5smTB1Ugy0sNqdXlChe/view?usp=sharing))
- **Content:** Player performance, demographics, and financial metrics
- **Key Features:**
  - Player information: Name, Age, Nationality, Club, Overall Rating, Potential, etc.
  - Financial metrics: Player value, Wage, Release Clause
  - Performance ratings: Various player positions and skills

## Key Predictions

- **Wages**
- **Release Clauses**
- **Player Positions**

## Challenges Addressed

- Handling complex, non-numeric data.
- Comprehensive data preprocessing and feature engineering.
- Developing robust models for prediction.

## Objective

The main goal of this project is to apply data preprocessing techniques, exploratory data analysis (EDA), and machine learning to build predictive models that can accurately forecast player wages, release clauses, and positions.

## Methodology

### Data Preprocessing:
- Handled missing values using mean replacement.
- Applied label encoding to convert categorical values.
- Removed outliers to improve model accuracy.
- Normalized and scaled features to ensure equal contribution to the model.

### Models:
1. **Regression Models**  
   - **RandomForestRegressor**: Used to predict player wages and release clauses. Achieved an R² score of 0.79 for wage prediction and 0.92 for release clauses.
   
2. **Classification Models**  
   - **RandomForestClassifier**: Achieved an accuracy of 80% in predicting player positions based on player attributes.

### Tools and Libraries:
- Python
- Jupyter Notebooks (Google Colab)
- Pandas, Numpy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit (for interactive web applications)
  


## Results

- **Wage Prediction (RandomForestRegressor):**
  - R² score: 0.79
  - RMSE: 11,247.78
  - MAE: 4,662.65

- **Release Clause Prediction (RandomForestRegressor):**
  - R² score: 0.92
  - RMSE: 3,509,909.37
  - MAE: 1,161,145.49

- **Position Prediction (RandomForestClassifier):**
  - Accuracy: 80%
  - F1-Score: 0.80 (weighted average)

## Conclusion

This project demonstrates the application of machine learning to predict critical metrics in football, including wages, release clauses, and player positions. The models developed provide actionable insights that can improve decision-making in football management.


## Deployment

To deploy the Streamlit app, use the following command:

```bash
streamlit run Home.py
