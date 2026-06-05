import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Simulate a realistic retail dataset
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="W")
stores = [1, 2, 3]
departments = [10, 20, 30]

data_rows = []
for date in dates:
    for store in stores:
        for dept in departments:
            # Base sales with seasonality (higher sales in Nov/Dec)
            base = 20000
            seasonality = (
                5000 if date.month in [11, 12] else np.random.randint(-2000, 2000)
            )
            markdown_effect = (
                np.random.choice([0, 1500], p=[0.7, 0.3])
                if date.month == 11
                else 0
            )
            sales = base + seasonality + markdown_effect + np.random.normal(0, 1000)

            # Introduce some random missingness in Markdown for cleaning practice
            markdown = (
                np.nan if np.random.rand() > 0.4 else np.random.uniform(500, 5000)
            )

            data_rows.append(
                {
                    "Date": date,
                    "Store": store,
                    "Dept": dept,
                    "Weekly_Sales": max(0, sales),
                    "IsHoliday": 1 if date.month == 12 and date.day > 20 else 0,
                    "Markdown": markdown,
                }
            )

df = pd.DataFrame(data_rows)

# 2. Data Cleaning Pipeline
df["Date"] = pd.to_datetime(df["Date"])
df["Markdown"] = df["Markdown"].fillna(0)

# 3. Exploratory Data Analysis & Visualizations
print("--- Dataset Summary ---")
print(df.info())
print("\n--- Descriptive Statistics ---")
print(df.describe())

# Visualization 1: Overall Sales Trend Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="Weekly_Sales", ci=None, color="teal")
plt.title("Aggregated Weekly Sales Trend (2024 - 2025)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales ($)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("sales_trend.png")
plt.close()

# Visualization 2: Department Performance Comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Dept", y="Weekly_Sales", palette="Set2")
plt.title("Sales Distribution Across Departments")
plt.xlabel("Department ID")
plt.ylabel("Weekly Sales ($)")
plt.savefig("dept_distribution.png")
plt.close()

# Save cleaned baseline data for the next phase
df.to_csv("cleaned_retail_data.csv", index=False)
print("\n[Success] EDA complete. Plots and 'cleaned_retail_data.csv' saved.")