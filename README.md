# Turbofan Engine Remaining Useful Life Prediction

## Project Overview

This project predicts the Remaining Useful Life (RUL) of turbofan engines using sensor measurements and machine learning techniques.

The project uses the NASA C-MAPSS FD001 dataset and applies data preprocessing, exploratory data analysis, feature engineering, rolling statistical features, and machine learning models to estimate the number of operational cycles remaining before engine failure.

## Objectives

- Analyze turbofan engine sensor data
- Calculate Remaining Useful Life (RUL)
- Remove constant and non-informative sensors
- Create rolling statistical features
- Train machine learning models
- Compare model performance using MAE and RMSE
- Generate final RUL predictions for test engines

## Dataset

The project uses the NASA C-MAPSS FD001 turbofan engine dataset.

The raw dataset files are intentionally excluded from this repository because of their size. The RUL reference files required for evaluation are included.

## Methodology

### 1. Data Preprocessing

- Loaded the FD001 training and test datasets
- Identified constant sensors
- Removed sensors that contained no useful variation
- Organized engine and cycle information

### 2. RUL Calculation

RUL was calculated for each engine cycle based on the remaining number of operational cycles before failure.

### 3. Feature Engineering

Rolling statistical features were created for important sensor measurements, including:

- Rolling mean
- Rolling standard deviation

A rolling window was used to capture recent sensor behavior and degradation patterns.

### 4. Machine Learning Models

The following models were trained and evaluated:

- Random Forest Regressor
- Gradient Boosting Regressor

### 5. Evaluation Metrics

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Results

### Validation Results

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 28.21 | 39.85 |
| Gradient Boosting | 28.37 | 39.44 |

### FD001 Test Results

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 20.27 | 27.06 |
| Gradient Boosting | 19.71 | 26.23 |

The test evaluation was performed using the final RUL values provided for the FD001 test engines.

## Important Features

The most influential features included:

- Cycle
- Sensor 15 rolling mean
- Sensor 4 rolling mean
- Sensor 11 rolling mean
- Sensor 9 rolling mean
- Sensor 2 rolling mean
- Sensor 14 rolling mean

The cycle number was the most important feature in the trained Random Forest model.

## Project Files

- eda.py - Exploratory data analysis
- week2_model.py - Week 2 modelling work
- week3_model.py - Week 3 feature engineering and model training
- week3_graphs.py - Visualization generation
- RUL_reader.py - RUL data reading utility
- week3_final_predictions.csv - Final test predictions
- week3_model_comparison.csv - Validation model comparison
- week3_test_model_comparison.csv - Test model comparison
- week3_feature_importance.csv - Feature importance results
- week3_actual_vs_predicted_RUL.png - Actual vs predicted RUL visualization
- week3_model_comparison_MAE.png - MAE comparison
- week3_model_comparison_RMSE.png - RMSE comparison
- week3_top15_feature_importance.png - Top feature importance visualization

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Machine Learning
- Regression
- Feature Engineering

## Conclusion

The project successfully developed a machine learning based approach for predicting the Remaining Useful Life of turbofan engines. Sensor data, cycle information, and rolling statistical features were used to train and evaluate regression models.

The final test evaluation demonstrated that the developed models can estimate the remaining operational cycles of the test engines with measurable predictive accuracy.

## Project Status

Completed.

