# Python Utility Functions

This Python file contains a set of utility functions for data processing, statistical analysis, memory optimization, and visualization. Below is the list of functions along with their descriptions and usage.

## Table of Contents
### Data Analysis
1. [seed_everything](#seed_everything)
2. [reduce_mem_usage](#reduce_mem_usage)
3. [info_miss_nunique](#info_miss_nunique)
4. [columns_info](#columns_info)
5. [confusion_table](#confusion_table)
6. [monte_carlo_chi2_test](#monte_carlo_chi2_test)
7. [cramers_v](#cramers_v)
8. [cramers_v_matrix](#cramers_v_matrix)
9. [categorical_vs_continuous_correlation](#categorical_vs_continuous_correlation)
10. [correlation_df](#correlation_df)
11. [correlation_target_df](#correlation_target_df)
12. [tests_for_normality](#tests_for_normality)
13. [continuous_vs_categorical_plots](#continuous_vs_categorical_plots)
14. [categorical_vs_categorical_plots](#categorical_vs_categorical_plots)
### Modeling
1. [model_dashboard_binary_classification](#model_dashboard_binary_classification)
 
---
### Data Analysis
#### seed_everything
- Ensures that all random processes are stable and deterministic by setting a seed for `random`, `os`, and `numpy` modules.

#### reduce_mem_usage
- Optimizes the memory usage of a pandas DataFrame by converting data types to more efficient types (e.g., `int8`, `float32`).

#### info_miss_nunique
- Analyzes missing values, data types, and the number of unique values in a DataFrame.

#### columns_info
- Categorizes the columns of a pandas DataFrame based on their data types and the presence of missing values.

#### confusion_table
- Generates a confusion matrix (cross-table) for two categorical features.

#### monte_carlo_chi2_test
- Calculate p-value for Chi2 test using Monte Carlo simulation.

#### cramers_v
- Computes Cramér's V statistic for the strength of association between two categorical variables.

#### cramers_v_matrix
- Creates a matrix of Cramér's V correlations for all pairs of categorical columns in a DataFrame.

#### categorical_vs_continuous_correlation
- Calculates the correlation (the square root of the R² score) between a categorical variable and a continuous variable using an OLS (Ordinary Least Squares) model.

#### correlation_df
- Calculate the Cramér's V correlation for all pairs of categorical columns, Pearson or Spearman correlation for all pairs of numeric columns and Sqrt of R^2 score for all pairs of numeric and categorical columns in a DataFrame.

#### correlation_target_df
- Calculate correlation between target and features only. The approach is like in [correlation_df](#correlation_df).

#### tests_for_normality
- Performs normality tests and optionally plots the distribution of the data.

#### continuous_vs_categorical_plots
- Plots a box plot and violin plot for the relationship between a continuous and categorical variable.

#### categorical_vs_categorical_plots
- Visualizes the relationship between two categorical variables using a stacked bar plot.

---
### Modeling 
#### model_dashboard_binary_classification
- Provides a detailed performance summary for a binary classification model. It calculates and displays key evaluation metrics such as accuracy, AUC for both PR-Recall and ROC curves, and Matthews Correlation Coefficient (MCC). It also prints a classification report (precision, recall, F1-score) and visualizes the performance through plots of the PR-Recall curve and Confusion Matrix.
---
Each function is documented using docstrings for clarity, and detailed explanations of the inputs, outputs, and functionality are provided.