import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load Cleaned Dataset
df = pd.read_csv("cleaned_retail_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# 2. Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Create lag features (previous week's sales within the same store/dept)
df = df.sort_values(by=["Store", "Dept", "Date"])
df["Lag_1_Sales"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
df["Lag_2_Sales"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(2)

# Drop rows where lag features are NaN due to shifting
df = df.dropna().reset_index(drop=True)

# 3. Train-Test Split (Chronological split to mimic real-world forecasting)
features = [
    "Store",
    "Dept",
    "IsHoliday",
    "Markdown",
    "Year",
    "Month",
    "Week",
    "Lag_1_Sales",
    "Lag_2_Sales",
]
target = "Weekly_Sales"

# Use the last 6 months of 2025 as the validation set
split_date = pd.to_datetime("2025-07-01")
train_df = df[df["Date"] < split_date]
test_df = df[df["Date"] >= split_date]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# 4. Model Training (LightGBM Regressor)
model = lgb.LGBMRegressor(
    n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42
)
model.fit(X_train, y_train)

# 5. Model Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared Score (R²): {r2:.4f}")

# 6. Save Forecast Results for Business Strategy
test_df = test_df.copy()
test_df["Predicted_Sales"] = predictions
test_df.to_csv("sales_forecast_predictions.csv", index=False)

# Feature Importance Plot
importance = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importance, color="slateblue")
plt.title("Feature Importance for Sales Forecasting")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("\n[Success] Forecasting complete. Predictions and metrics saved.")