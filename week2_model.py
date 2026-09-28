import pandas as pd
# Constant sensors identified during Week 1
constant_sensors = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19"
]
# Load the cleaned dataset from Week 1
data = pd.read_csv("train_FD001_clean.csv")

print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumns:")
print(data.columns.tolist())

# Separate input features and target
X = data.drop(columns=["RUL"])
y = data["RUL"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nRUL summary:")
print(y.describe())

from sklearn.model_selection import train_test_split

# Get unique engine IDs
engine_ids = data["engine_id"].unique()

# Split engines into training and validation groups
train_engines, val_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

# Create training and validation datasets
train_data = data[data["engine_id"].isin(train_engines)]
val_data = data[data["engine_id"].isin(val_engines)]

print("\nTraining engines:", len(train_engines))
print("Validation engines:", len(val_engines))

print("\nTraining data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)

# Separate features and target
# engine_id is only an identifier, so it is not used as a model feature

features = [
    col for col in data.columns
    if col not in ["engine_id", "RUL"]
]

X_train = train_data[features]
y_train = train_data["RUL"]

X_val = val_data[features]
y_val = val_data["RUL"]

print("\nFeatures used for modeling:")
print(features)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Create the Random Forest regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions on validation data
y_pred = model.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)

# Feature importance analysis
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

from sklearn.ensemble import GradientBoostingRegressor

# Create Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

print("\nTraining Gradient Boosting model...")

# Train the model
gb_model.fit(X_train, y_train)

print("Gradient Boosting training completed!")

# Make predictions
gb_pred = gb_model.predict(X_val)

# Evaluate the model
gb_mae = mean_absolute_error(y_val, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))

print("\nGradient Boosting Evaluation:")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)

# Compare model performance
results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting"],
    "MAE": [mae, gb_mae],
    "RMSE": [rmse, gb_rmse]
})

print("\nModel Comparison:")
print(results)

# Save model comparison results
results.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved as model_comparison.csv")

# ==========================================
# Test Dataset Prediction
# ==========================================

# Column names for the original FD001 test dataset
columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4",
    "sensor_5", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_10", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_16",
    "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

# Load test dataset
test_data = pd.read_csv(
    "test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Load true RUL values for the test engines
true_rul = pd.read_csv(
    "RUL_FD001.txt",
    sep=r"\s+",
    header=None
)

# Name the RUL column
true_rul.columns = ["RUL"]

print("\nTest dataset shape:", test_data.shape)
print("True RUL shape:", true_rul.shape)

# Remove the same constant sensors removed during Week 1
test_clean = test_data.drop(columns=constant_sensors)

print("\nClean test dataset shape:", test_clean.shape)

# Keep only the same modeling features used during training
X_test = test_clean[features]

print("\nTest feature shape:", X_test.shape)

# Predict RUL for every test observation
test_predictions = gb_model.predict(X_test)

print("\nFirst 10 test predictions:")
print(test_predictions[:10])

# ==========================================
# Final RUL Prediction for Each Test Engine
# ==========================================

# Add predictions to the test dataset
test_clean["predicted_RUL"] = test_predictions

# Get the last observed cycle for each test engine
final_predictions = (
    test_clean
    .sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .copy()
)

# Keep only the required columns
final_predictions = final_predictions[
    ["engine_id", "cycle", "predicted_RUL"]
]

# Add the true RUL values
final_predictions["true_RUL"] = true_rul["RUL"].values

# Calculate prediction error
final_predictions["error"] = (
    final_predictions["predicted_RUL"]
    - final_predictions["true_RUL"]
)

final_predictions["absolute_error"] = (
    final_predictions["error"].abs()
)

print("\nFinal Test Engine Predictions:")
print(final_predictions.head(10))

# Calculate final test metrics
test_mae = mean_absolute_error(
    final_predictions["true_RUL"],
    final_predictions["predicted_RUL"]
)

test_rmse = np.sqrt(
    mean_squared_error(
        final_predictions["true_RUL"],
        final_predictions["predicted_RUL"]
    )
)

print("\nFinal Test Evaluation:")
print("MAE:", test_mae)
print("RMSE:", test_rmse)

# Save final predictions
final_predictions.to_csv(
    "FD001_final_predictions.csv",
    index=False
)

print("\nFinal predictions saved as FD001_final_predictions.csv")

# ==========================================
# Actual vs Predicted RUL Visualization
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    final_predictions["engine_id"],
    final_predictions["true_RUL"],
    marker="o",
    label="Actual RUL"
)

plt.plot(
    final_predictions["engine_id"],
    final_predictions["predicted_RUL"],
    marker="x",
    label="Predicted RUL"
)

plt.xlabel("Engine ID")
plt.ylabel("Remaining Useful Life (Cycles)")
plt.title("Actual vs Predicted RUL - FD001 Test Engines")
plt.legend()
plt.grid()

plt.savefig(
    "actual_vs_predicted_RUL.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nActual vs Predicted RUL graph saved as actual_vs_predicted_RUL.png")