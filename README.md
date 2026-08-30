# Logistics Data Analysis

## Week 1: Strategic Planning and Data Exploration in Logistics
This project focuses on improving last-mile logistics operations using data science and Python.

### Project Objective
The main objective is to analyze logistics data and identify factors affecting:
- Delivery delays
- Delivery costs
- Route efficiency
- Vehicle allocation
- Customer satisfaction

### Key Performance Indicators
- On-Time Delivery Rate
- Average Delivery Delay
- Delivery Cost per Shipment
- Average Delivery Time
- Customer Delivery Rating

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google OR-Tools

### Data Science Approach
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Predictive Modeling
6. Clustering
7. Route Optimization
8. Performance Evaluation

### Proposed Techniques
- Regression
- Classification
- K-Means Clustering
- Vehicle Route Optimization
- Exploratory Data Analysis

---

## Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
- **`src/generate_dataset.py`** — simulates raw data collection, producing
  `data/delivery_logistics_raw.csv` (500 shipment records with intentionally
  injected missing values, duplicates, outliers, and inconsistent text entries).
- **`src/data_preprocessing.py`** — the cleaning pipeline: initial inspection,
  duplicate removal, categorical standardization, missing-value imputation
  (median for numeric fields, mode for categorical fields), IQR-based outlier
  detection and capping, Min-Max and Z-score normalization, and light feature
  engineering. Produces `data/delivery_logistics_cleaned.csv`.
- **`Week_2_Report.docx`** — full write-up of methodology, code snippets, and
  a reflection on how data quality affects logistics decision-making.

```bash
cd src
python generate_dataset.py       # creates data/delivery_logistics_raw.csv
python data_preprocessing.py     # creates data/delivery_logistics_cleaned.csv
```

---

## Week 3: Advanced Data Analysis and Visualization in Logistics
- **`src/eda_visualization.py`** — loads the Week 2 cleaned dataset, runs
  descriptive statistics and correlation analysis, corrects a residual
  data-quality issue found during EDA (a few non-physical negative
  `distance_km` values left over from Week 2's IQR capping), and produces
  seven visualizations (heatmap, histogram, boxplot, scatter plot, and bar/
  line charts) saved to `visuals/`.
- **`Week_3_Report.docx`** — full write-up with embedded charts, methodology,
  chart-choice justification, and analytical insights/recommendations.

```bash
cd src
python eda_visualization.py      # creates visuals/*.png and data/summary_statistics.csv
```

## Project Structure
```
logistics-data-analysis/
│
├── README.md
├── Week_1_Report.docx
├── Week_2_Report.docx
├── Week_3_Report.docx
│
├── data/
│   ├── README.md
│   ├── delivery_logistics_raw.csv
│   ├── delivery_logistics_cleaned.csv
│   └── summary_statistics.csv
│
├── visuals/
│   ├── 01_correlation_heatmap.png
│   ├── 02_delivery_time_distribution.png
│   ├── 03_cost_by_vehicle_boxplot.png
│   ├── 04_distance_vs_time_scatter.png
│   ├── 05_avg_cost_by_region.png
│   ├── 06_delay_rate_by_day.png
│   └── 07_rating_by_weather.png
│
└── src/
    ├── logistics_analysis.py
    ├── generate_dataset.py
    ├── data_preprocessing.py
    └── eda_visualization.py
```
