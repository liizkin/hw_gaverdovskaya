## 1. Project Description

This program is designed to analyze mobile phone pricing data from various companies.

## 2. Technology Stack
Language: Python 3.13

Libraries:

- matplotlib, seaborn (data visualization)

- pandas, numpy (data processing)

## 3. Data Analysis
The following tasks were performed (code located in `eda_analysis.py`):

1. Output shape, data types, missing values, and duplicates.
2. For each feature, use `describe()` and display the distribution by `price_range` classes.
3. Create a correlation heatmap (save to `output/corr.png`).
4. Create boxplots for key features (`ram`, `battery_power`, `px_height`) broken down by `price_range` (`output/boxplots.png`).
5. Provide a textual summary: identify the 5 features most strongly associated with the price segment and explain why.

Input data is located in `train.csv`; logs are in `logs.txt`.

Output data is available in the `/output` directory:

- [Boxplots of key features](doc/boxplot.png)
- [Correlation matrix](doc/corr.png)
- [Descriptive statistics for features](doc/descriptive_stats.csv)
- [Descriptive statistics for features by price_range](doc/descriptive_stats_by_price_range.csv)
- [Textual summary](doc/eda_summary.json)
