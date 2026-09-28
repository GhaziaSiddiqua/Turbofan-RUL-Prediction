import pandas as pd
import numpy as np

# --------------------------------------------------
# PROJECT 6 - WEEK 3
# Feature Engineering for RUL Prediction
# --------------------------------------------------

print("Loading FD001 training dataset...")

# Column names for FD001
columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

# Load training data
train_data = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Original dataset shape:", train_data.shape)

# --------------------------------------------------
# Remove constant sensors
# --------------------------------------------------

constant_sensors = []

for column in train_data.columns:
    if train_data[column].nunique() <= 1:
        constant_sensors.append(column)

print("\nConstant sensors:")
print(constant_sensors)

train_data = train_data.drop(columns=constant_sensors)

print("\nShape after removing constant sensors:")
print(train_data.shape)

# --------------------------------------------------
# Calculate RUL
# --------------------------------------------------

max_cycles = train_data.groupby("engine_id")["cycle"].max()

train_data["max_cycle"] = train_data["engine_id"].map(max_cycles)

train_data["RUL"] = (
    train_data["max_cycle"] - train_data["cycle"]
)

train_data = train_data.drop(columns=["max_cycle"])

print("\nRUL created successfully.")

print("\nRUL summary:")
print(train_data["RUL"].describe())

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

print("\nCreating rolling features...")

# Sensors that were useful in Week 2
selected_sensors = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

# Make sure selected sensors exist
selected_sensors = [
    sensor for sensor in selected_sensors
    if sensor in train_data.columns
]

# Rolling mean over previous/current cycles
for sensor in selected_sensors:
    train_data[f"{sensor}_rolling_mean"] = (
        train_data
        .groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )

# Rolling standard deviation
for sensor in selected_sensors:
    train_data[f"{sensor}_rolling_std"] = (
        train_data
        .groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).std())
    )

# Replace NaN values created by rolling standard deviation
train_data = train_data.fillna(0)

print("Rolling features created successfully.")

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nNew dataset shape:", train_data.shape)

print("\nNew feature columns:")

for column in train_data.columns:
    print(column)

print("\nFirst 5 rows with engineered features:")
print(train_data.head())

# --------------------------------------------------
# Save Week 3 processed dataset
# --------------------------------------------------

train_data.to_csv(
    "week3_feature_engineered.csv",
    index=False
)

print("\nFeature-engineered dataset saved as:")
print("week3_feature_engineered.csv")

# --------------------------------------------------
# WEEK 3 - MODEL TRAINING
# --------------------------------------------------

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("\n" + "=" * 60)
print("WEEK 3 MODEL TRAINING")
print("=" * 60)

# --------------------------------------------------
# Create engine-wise train/validation split
# --------------------------------------------------

engine_ids = train_data["engine_id"].unique()

np.random.seed(42)
np.random.shuffle(engine_ids)

train_engines = engine_ids[:80]
val_engines = engine_ids[80:]

train_df = train_data[
    train_data["engine_id"].isin(train_engines)
].copy()

val_df = train_data[
    train_data["engine_id"].isin(val_engines)
].copy()

print("\nTraining engines:", len(train_engines))
print("Validation engines:", len(val_engines))

print("Training data shape:", train_df.shape)
print("Validation data shape:", val_df.shape)

# --------------------------------------------------
# Select features
# --------------------------------------------------

features = [
    column for column in train_data.columns
    if column not in ["engine_id", "RUL"]
]

X_train = train_df[features]
y_train = train_df["RUL"]

X_val = val_df[features]
y_val = val_df["RUL"]

print("\nNumber of features used:", len(features))

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# --------------------------------------------------
# Random Forest
# --------------------------------------------------

print("\nTraining Week 3 Random Forest model...")

rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_val)

rf_mae = mean_absolute_error(y_val, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predictions))

print("Random Forest training completed!")

print("\nRandom Forest Evaluation:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)

# --------------------------------------------------
# Gradient Boosting
# --------------------------------------------------

print("\nTraining Week 3 Gradient Boosting model...")

gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_val)

gb_mae = mean_absolute_error(y_val, gb_predictions)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_predictions))

print("Gradient Boosting training completed!")

print("\nGradient Boosting Evaluation:")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)

# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Week 3 Random Forest",
        "Week 3 Gradient Boosting"
    ],
    "MAE": [
        rf_mae,
        gb_mae
    ],
    "RMSE": [
        rf_rmse,
        gb_rmse
    ]
})

print("\n" + "=" * 60)
print("WEEK 3 MODEL COMPARISON")
print("=" * 60)

print(comparison)

comparison.to_csv(
    "week3_model_comparison.csv",
    index=False
)

print("\nModel comparison saved as:")
print("week3_model_comparison.csv")

# --------------------------------------------------
# Feature Importance - Gradient Boosting
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": gb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

feature_importance.to_csv(
    "week3_feature_importance.csv",
    index=False
)

print("\nFeature importance saved as:")
print("week3_feature_importance.csv")

print("\nWeek 3 model training completed successfully!")

# ============================================================
# WEEK 3 TEST DATA EVALUATION
# ============================================================

print("\n============================================================")
print("WEEK 3 TEST DATA EVALUATION")
print("============================================================")

# Load FD001 test dataset
print("\nLoading FD001 test dataset...")

test_data = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None)

test_data.columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

true_rul = pd.read_csv(
    "RUL_FD001.txt",
    sep=r"\s+",
    header=None
)

# Load true RUL values
true_rul = pd.read_csv("RUL_FD001.txt", sep=r"\s+", header=None)

print("Test dataset shape:", test_data.shape)
print("True RUL shape:", true_rul.shape)


# ------------------------------------------------------------
# Remove the same constant sensors used during training
# ------------------------------------------------------------

test_data = test_data.drop(columns=constant_sensors)

print("\nTest data shape after removing constant sensors:")
print(test_data.shape)


# ------------------------------------------------------------
# Create rolling features for test data
# ------------------------------------------------------------

print("\nCreating rolling features for test data...")

rolling_sensors = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

window = 5

for sensor in rolling_sensors:

    test_data[sensor + "_rolling_mean"] = (
        test_data.groupby("engine_id")[sensor]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    test_data[sensor + "_rolling_std"] = (
        test_data.groupby("engine_id")[sensor]
        .rolling(window=window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )


print("Rolling features created successfully!")


# ------------------------------------------------------------
# Select the same features used during Week 3 training
# ------------------------------------------------------------

# Use the exact same feature names and order as training
feature_columns = X_train.columns.tolist()

X_test = test_data[feature_columns]

print("\nTest feature shape:", X_test.shape)


# ------------------------------------------------------------
# Make predictions using Week 3 Random Forest
# ------------------------------------------------------------

print("\nGenerating Random Forest predictions...")

rf_test_predictions = rf_model.predict(X_test)

print("Random Forest predictions generated!")


# ------------------------------------------------------------
# Make predictions using Week 3 Gradient Boosting
# ------------------------------------------------------------

print("\nGenerating Gradient Boosting predictions...")

gb_test_predictions = gb_model.predict(X_test)

print("Gradient Boosting predictions generated!")


# ------------------------------------------------------------
# Select the final prediction for each engine
# ------------------------------------------------------------

# The test dataset contains multiple rows for each engine.
# We use the last cycle of each engine because it represents
# the final observed condition before prediction.

last_cycle_indices = test_data.groupby("engine_id")["cycle"].idxmax()

rf_final_predictions = rf_test_predictions[
    last_cycle_indices.to_numpy()
]

gb_final_predictions = gb_test_predictions[
    last_cycle_indices.to_numpy()
]


# ------------------------------------------------------------
# Prepare actual RUL values
# ------------------------------------------------------------

actual_rul = true_rul.iloc[:, 0].values

print("\nNumber of actual RUL values:", len(actual_rul))
print("Number of test engine predictions:", len(rf_final_predictions))


# ------------------------------------------------------------
# Evaluate Random Forest
# ------------------------------------------------------------

rf_test_mae = mean_absolute_error(
    actual_rul,
    rf_final_predictions
)

rf_test_rmse = np.sqrt(
    mean_squared_error(
        actual_rul,
        rf_final_predictions
    )
)

print("\nRandom Forest Test Evaluation:")
print("MAE:", rf_test_mae)
print("RMSE:", rf_test_rmse)


# ------------------------------------------------------------
# Evaluate Gradient Boosting
# ------------------------------------------------------------

gb_test_mae = mean_absolute_error(
    actual_rul,
    gb_final_predictions
)

gb_test_rmse = np.sqrt(
    mean_squared_error(
        actual_rul,
        gb_final_predictions
    )
)

print("\nGradient Boosting Test Evaluation:")
print("MAE:", gb_test_mae)
print("RMSE:", gb_test_rmse)


# ------------------------------------------------------------
# Model comparison on test data
# ------------------------------------------------------------

test_comparison = pd.DataFrame({
    "Model": [
        "Week 3 Random Forest",
        "Week 3 Gradient Boosting"
    ],
    "MAE": [
        rf_test_mae,
        gb_test_mae
    ],
    "RMSE": [
        rf_test_rmse,
        gb_test_rmse
    ]
})

print("\n============================================================")
print("WEEK 3 TEST MODEL COMPARISON")
print("============================================================")

print(test_comparison)


# ------------------------------------------------------------
# Create final prediction table
# ------------------------------------------------------------

final_predictions = pd.DataFrame({
    "engine_id": sorted(test_data["engine_id"].unique()),
    "actual_RUL": actual_rul,
    "Random_Forest_Prediction": rf_final_predictions,
    "Gradient_Boosting_Prediction": gb_final_predictions
})


# Calculate errors for Gradient Boosting
final_predictions["GB_error"] = (
    final_predictions["Gradient_Boosting_Prediction"]
    - final_predictions["actual_RUL"]
)

final_predictions["GB_absolute_error"] = (
    final_predictions["GB_error"].abs()
)


print("\nFirst 10 final predictions:")
print(final_predictions.head(10))


# ------------------------------------------------------------
# Save final predictions
# ------------------------------------------------------------

final_predictions.to_csv(
    "week3_final_predictions.csv",
    index=False
)

test_comparison.to_csv(
    "week3_test_model_comparison.csv",
    index=False
)

print("\nFinal predictions saved as:")
print("week3_final_predictions.csv")

print("\nTest model comparison saved as:")
print("week3_test_model_comparison.csv")

print("\n============================================================")
print("WEEK 3 TEST EVALUATION COMPLETED SUCCESSFULLY!")
print("============================================================")