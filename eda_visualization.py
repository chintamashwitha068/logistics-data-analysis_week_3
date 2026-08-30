"""
eda_visualization.py
----------------------
Week 3: Advanced Data Analysis and Visualization in Logistics

Loads the cleaned dataset produced in Week 2
(data/delivery_logistics_cleaned.csv) and performs exploratory data
analysis (EDA): descriptive statistics, distributions, correlations,
and a set of visualizations that surface operational patterns in the
logistics data (delivery time, cost, distance, vehicle type, weather,
region).

All charts are saved as PNG files into ../visuals/ so they can be
embedded in the Week 3 Word report.

Run from the src/ directory:
    python eda_visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT = "../visuals/"

df = pd.read_csv("../data/delivery_logistics_cleaned.csv", parse_dates=["order_date"])

print("=" * 70)
print("STEP 1: DATASET OVERVIEW")
print("=" * 70)
print("Shape:", df.shape)
print(df.dtypes)

# ---------------------------------------------------------------------
# STEP 1.5: DATA VALIDATION REFINEMENT (found during EDA)
# ---------------------------------------------------------------------
# The Week 2 IQR-based capping bounds a column at [Q1-1.5*IQR, Q3+1.5*IQR]
# statistically, without knowing the column's real-world constraints.
# For distance_km this produced a lower bound below zero, so a handful
# of shipments were capped to a NEGATIVE distance -- impossible in
# reality, and it also breaks the derived cost_per_km feature (division
# by a small/negative number produces extreme, misleading ratios).
# This is corrected here by flooring distance_km at a realistic minimum
# (0.5 km) before any distance-based analysis, and cost_per_km is
# recomputed on the corrected column.
neg_dist = (df["distance_km"] < 0.5).sum()
print(f"\nSTEP 1.5: Found {neg_dist} shipments with non-physical distance_km "
      f"(< 0.5 km, including negative values) left over from Week 2 IQR "
      f"capping. Flooring these at 0.5 km before analysis.")
df["distance_km"] = df["distance_km"].clip(lower=0.5)
df["cost_per_km"] = (df["delivery_cost"] / df["distance_km"]).round(2)

# ---------------------------------------------------------------------
# STEP 2: DESCRIPTIVE STATISTICS (central tendency & spread)
# ---------------------------------------------------------------------
numeric_cols = ["distance_km", "delivery_time_hr", "delivery_cost", "customer_rating", "cost_per_km"]
desc = df[numeric_cols].describe().T
desc["skew"] = df[numeric_cols].skew()
print("\nSTEP 2: Descriptive statistics (mean, std, quartiles, skew)")
print(desc)
desc.to_csv("../data/summary_statistics.csv")

# ---------------------------------------------------------------------
# STEP 3: CORRELATION ANALYSIS
# ---------------------------------------------------------------------
corr = df[numeric_cols].corr()
print("\nSTEP 3: Correlation matrix")
print(corr.round(2))

plt.figure(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Key Logistics Metrics")
plt.tight_layout()
plt.savefig(OUT + "01_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------------
# STEP 4: DISTRIBUTION ANALYSIS
# ---------------------------------------------------------------------
# Histogram: delivery time distribution
plt.figure(figsize=(7, 4.5))
sns.histplot(df["delivery_time_hr"], bins=25, kde=True, color="#2E75B6")
plt.axvline(df["delivery_time_hr"].median(), color="red", linestyle="--", label="Median")
plt.title("Distribution of Delivery Time (hours)")
plt.xlabel("Delivery Time (hr)")
plt.ylabel("Number of Shipments")
plt.legend()
plt.tight_layout()
plt.savefig(OUT + "02_delivery_time_distribution.png")
plt.close()

# Boxplot: delivery cost by vehicle type (distribution + outlier view)
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="vehicle_type", y="delivery_cost", palette="Blues")
plt.title("Delivery Cost Distribution by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Delivery Cost")
plt.tight_layout()
plt.savefig(OUT + "03_cost_by_vehicle_boxplot.png")
plt.close()

# ---------------------------------------------------------------------
# STEP 5: RELATIONSHIP ANALYSIS
# ---------------------------------------------------------------------
# Scatter: distance vs delivery time, colored by weather condition
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="distance_km", y="delivery_time_hr", hue="weather_condition",
                 palette="Set2", alpha=0.75)
plt.title("Distance vs Delivery Time, by Weather Condition")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hr)")
plt.tight_layout()
plt.savefig(OUT + "04_distance_vs_time_scatter.png")
plt.close()

# ---------------------------------------------------------------------
# STEP 6: CATEGORICAL / OPERATIONAL BREAKDOWNS
# ---------------------------------------------------------------------
# Bar chart: average delivery cost per region
region_cost = df.groupby("region")["delivery_cost"].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 4.5))
sns.barplot(x=region_cost.index, y=region_cost.values, palette="Blues_d")
plt.title("Average Delivery Cost by Region")
plt.xlabel("Region")
plt.ylabel("Average Delivery Cost")
plt.tight_layout()
plt.savefig(OUT + "05_avg_cost_by_region.png")
plt.close()

# Line chart: on-time vs delayed shipments by day of week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
delay_by_day = df.groupby("order_day_of_week")["is_delayed"].mean().reindex(day_order)
plt.figure(figsize=(7, 4.5))
plt.plot(delay_by_day.index, delay_by_day.values, marker="o", color="#C00000")
plt.title("Delay Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Proportion of Delayed Shipments")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT + "06_delay_rate_by_day.png")
plt.close()

# Bar chart: average customer rating by weather condition
weather_rating = df.groupby("weather_condition")["customer_rating"].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 4.5))
sns.barplot(x=weather_rating.index, y=weather_rating.values, palette="Greens_d")
plt.title("Average Customer Rating by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Average Customer Rating (1-5)")
plt.tight_layout()
plt.savefig(OUT + "07_rating_by_weather.png")
plt.close()

print("\nSTEP 6: All visualizations saved to visuals/ directory")

# ---------------------------------------------------------------------
# STEP 7: KEY INSIGHT SUMMARY (printed for the report narrative)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: KEY NUMERIC INSIGHTS")
print("=" * 70)
print("Overall delay rate:", round(df["is_delayed"].mean(), 2))
print("Correlation distance vs delivery time:", round(corr.loc["distance_km", "delivery_time_hr"], 2))
print("Correlation delivery time vs cost:", round(corr.loc["delivery_time_hr", "delivery_cost"], 2))
print("Highest average cost region:", region_cost.idxmax(), "(", round(region_cost.max(), 2), ")")
print("Worst delay day:", delay_by_day.idxmax(), "(", round(delay_by_day.max(), 2), ")")
print("Lowest-rated weather condition:", weather_rating.idxmin(), "(", round(weather_rating.min(), 2), ")")
