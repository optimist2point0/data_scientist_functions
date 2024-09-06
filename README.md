# Python Utility Functions

This Python file contains a set of utility functions for data processing, statistical analysis, memory optimization, and visualization. Below is the list of functions along with their descriptions and usage.

## Table of Contents
### Data Analysis
1. [seed_everything](#seed_everything)
2. [reduce_mem_usage](#reduce_mem_usage)
3. [confusion_table](#confusion_table)
4. [cramers_v](#cramers_v)
5. [cramers_v_matrix](#cramers_v_matrix)
6. [info_miss_nunique](#info_miss_nunique)
7. [tests_for_normality](#tests_for_normality)
8. [continuous_vs_categorical_plots](#continuous_vs_categorical_plots)
9. [categorical_vs_categorical_plots](#categorical_vs_categorical_plots)
### Modeling
1. [model_dashboard_binary_classification](#model_dashboard_binary_classification)
 
---
### Data Analysis
#### seed_everything
- Ensures that all random processes are stable and deterministic by setting a seed for `random`, `os`, and `numpy` modules.

#### reduce_mem_usage
- Optimizes the memory usage of a pandas DataFrame by converting data types to more efficient types (e.g., `int8`, `float32`).

#### confusion_table
- Generates a confusion matrix (cross-table) for two categorical features.

#### cramers_v
- Computes Cramér's V statistic for the strength of association between two categorical variables.

#### cramers_v_matrix
- Creates a matrix of Cramér's V correlations for all pairs of categorical columns in a DataFrame.

#### info_miss_nunique
- Analyzes missing values, data types, and the number of unique values in a DataFrame.

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