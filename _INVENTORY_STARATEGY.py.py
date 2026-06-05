import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load Predictions Matrix
df_preds = pd.read_csv("sales_forecast_predictions.csv")

# 2. Operational Inventory Logic
# Suppose the safety stock multiplier is 1.2x the predicted weekly demand
df_preds["Target_Inventory_Units"] = np.ceil(df_preds["Predicted_Sales"] / 50 * 1.2)

# Simulate actual current warehouse inventory for comparison
np.random.seed(10)
df_preds["Current_Inventory_Units"] = np.random.randint(
    300, 700, size=len(df_preds)
)

# Identify stock status
df_preds["Inventory_Gap"] = (
    df_preds["Current_Inventory_Units"] - df_preds["Target_Inventory_Units"]
)


def determine_action(gap):
    if gap < -50:
        return "Urgent Restock"
    elif gap > 150:
        return "Overstocked / Clearance Run"
    else:
        return "Optimal"


df_preds["Action_Required"] = df_preds["Inventory_Gap"].apply(determine_action)

# 3. Generate Business Insights Report
print("--- Inventory Optimization Summary ---")
print(df_preds["Action_Required"].value_counts())

# Save finalized operational report
df_preds.to_csv("final_inventory_action_plan.csv", index=False)

# 4. Action Item Visualization
plt.figure(figsize=(8, 5))
df_preds["Action_Required"].value_counts().plot(
    kind="bar", color=["g", "orange", "r"], alpha=0.8
)
plt.title("Warehouse Inventory Action Flags")
plt.xlabel("Action Category")
plt.ylabel("Number of Store-Department Units")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("inventory_action_flags.png")
plt.close()
print(
    "\n[Success] Inventory optimization executed. Final action plan saved as CSV."
)