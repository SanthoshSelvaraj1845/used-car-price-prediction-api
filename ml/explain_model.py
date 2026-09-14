import joblib
import pandas as pd
import matplotlib.pyplot as plt


# Load trained pipeline
model = joblib.load("ml/saved_model/model.joblib")


# Get preprocessing and Random Forest model
preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["regressor"]


# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()


# Get feature importance
importances = random_forest.feature_importances_


# Create DataFrame
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})


# Sort by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


print("\nFeature Importance:")
print(importance_df)


# Plot top 15 features
top_features = importance_df.head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Used Car Price Prediction - Feature Importance")

plt.tight_layout()

plt.savefig(
    "ml/saved_model/feature_importance.png"
)

plt.show()