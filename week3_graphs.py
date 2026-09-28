import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# WEEK 3 GRAPHS
# ============================================================

print("Loading Week 3 results...")

# ------------------------------------------------------------
# Load model comparison results
# ------------------------------------------------------------

model_comparison = pd.read_csv("week3_test_model_comparison.csv")

print("\nModel comparison:")
print(model_comparison)


# ------------------------------------------------------------
# Load final predictions
# ------------------------------------------------------------

predictions = pd.read_csv("week3_final_predictions.csv")

print("\nPrediction data loaded!")
print("Number of test engines:", len(predictions))


# ============================================================
# GRAPH 1: ACTUAL RUL VS GRADIENT BOOSTING PREDICTIONS
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    predictions["engine_id"],
    predictions["actual_RUL"],
    label="Actual RUL",
    linewidth=2
)

plt.plot(
    predictions["engine_id"],
    predictions["Gradient_Boosting_Prediction"],
    label="Gradient Boosting Prediction",
    linewidth=2
)

plt.xlabel("Engine ID")
plt.ylabel("Remaining Useful Life (Cycles)")
plt.title("Actual RUL vs Gradient Boosting Predicted RUL")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "week3_actual_vs_predicted_RUL.png",
    dpi=300
)

plt.show()

print("\nGraph 1 saved:")
print("week3_actual_vs_predicted_RUL.png")


# ============================================================
# GRAPH 2: MODEL COMPARISON - MAE
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["MAE"]
)

plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (Cycles)")
plt.title("Week 3 Model Comparison - MAE")

plt.tight_layout()

plt.savefig(
    "week3_model_comparison_MAE.png",
    dpi=300
)

plt.show()

print("\nGraph 2 saved:")
print("week3_model_comparison_MAE.png")


# ============================================================
# GRAPH 3: MODEL COMPARISON - RMSE
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["RMSE"]
)

plt.xlabel("Model")
plt.ylabel("Root Mean Squared Error (Cycles)")
plt.title("Week 3 Model Comparison - RMSE")

plt.tight_layout()

plt.savefig(
    "week3_model_comparison_RMSE.png",
    dpi=300
)

plt.show()

print("\nGraph 3 saved:")
print("week3_model_comparison_RMSE.png")


# ============================================================
# GRAPH 4: TOP 15 FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.read_csv(
    "week3_feature_importance.csv"
)

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Important Features - Random Forest")

plt.tight_layout()

plt.savefig(
    "week3_top15_feature_importance.png",
    dpi=300
)

plt.show()

print("\nGraph 4 saved:")
print("week3_top15_feature_importance.png")


# ============================================================
# COMPLETED
# ============================================================

print("\n============================================================")
print("ALL WEEK 3 GRAPHS GENERATED SUCCESSFULLY!")
print("============================================================")