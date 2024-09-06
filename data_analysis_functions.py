import os
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display_functions import display

from scipy.stats import chi2_contingency, shapiro, norm, normaltest, anderson


# Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    """
    Make all random process stable.

    Args:
        seed (int): seed for randomness (default 0)
    Returns:
        fixed random processes
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by pd.DataFrame by converting data types

    Args:
        df (pd.DataFrame, pd.Series): DataFrame for memory usage reducing
        verbose (boll): bool values for printing
    Returns:
         pd.DataFrame: The reduced df.
    """
    numerics = ['int16', 'int32', 'int64', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem)
        )
    return df


# Cross-table for Cramers' V correlation
def confusion_table(feature_1, feature_2):
    """
    Calculate cross-table (confusion table) for Chi2-test.

    Args:
        feature_1: array-like of shape (n, ) with values to calculate cross-table
        feature_2: array-like of shape (n, ) with values to calculate cross-table
    Returns:
        table (pd.DataFrame): cross-table
        lam (int): lambda_ parameter determines Chi2-test (1) or G-test (0)
        text (None or str): None or text of WARNING
    """

    table = pd.crosstab(feature_1, feature_2)
    lam = 1
    text = None

    if (table < 5).any().any():
        text = "WARNING! There are cells with frequencies < 5. G-test will be used instead of Chi2."
        lam = 0
    else:
        pass
    return table, lam, text


# Cramers' V correlation for cat. col.
def cramers_v(cross_table, lam):
    """
    Calculate Cramér's V statistic for categorical-categorical association.

    Args:
        cross_table (pd.DataFrame): cross-table of 2 variables
        lam (int): lambda_ parameter determines Chi2-test (1) or G-test (0)
    Returns:
        v (float): Cramers' V correlation
    """
    # Chi2_contingency returns the chi2 value, p-value, degrees of freedom, and expected frequencies
    chi2, _, _, _ = chi2_contingency(cross_table, lambda_=lam)

    # Get the total number of observations
    n = cross_table.sum().sum()

    # Get the number of rows and columns in the confusion matrix
    k, r = cross_table.shape

    # Calculate Cramér's V
    v = np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

    return v


# Analogy to pd.DataFrame.corr() function for cat. col.
def cramers_v_matrix(df, cat_cols):
    """
    Calculate the Cramér's V matrix for all pairs of categorical columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to calculate Cramers' V correlation values
        cat_cols (list): list of categorical columns names
    Returns:
        cramers_v_df (pd.DataFrame): DataFrame of Cramers' V correlation values (like df.corr())
    """
    # Initialize an empty DataFrame to store Cramér's V values
    n = len(cat_cols)
    cramers_v_df = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)

    count = 0

    # Iterate over all pairs of columns
    for i in range(n):
        for j in range(i, n):

            # Create a contingency table for the pair of columns
            table, lam, text = confusion_table(df[cat_cols[i]], df[cat_cols[j]])

            if text and count < 4:
                count += 1
                print(text)

            # Calculate Cramér's V for this pair
            cramers_v_value = cramers_v(table, lam)

            cramers_v_df.iat[i, j] = cramers_v_value
            cramers_v_df.iat[j, i] = cramers_v_value

    # G-test don't put 1 on diagonal
    for i in range(cramers_v_df.shape[0]):
        cramers_v_df.iloc[i, i] = 1

    return cramers_v_df


# Missed values info
def info_miss_nunique(df, displ=True, verbose=True):
    """
    For DataFrame calculate number/percent of missed values, dtype and number of unique values per column.

    Args:
        df (pd.DataFrame): DataFrame to inspect
        displ (bool): True for display result
        verbose (boll): True for print suggestions

    """
    missed_info = df.isnull().sum().to_frame('# of Missed')
    missed_info['% of Missed'] = ((missed_info['# of Missed'] / df.shape[0]) * 100).round(2)
    missed_info['Type'] = df.dtypes
    missed_info['N_unique'] = df.nunique()
    missed_info = missed_info[['Type', 'N_unique', '# of Missed', '% of Missed']]
    missed_info.sort_values(by=['# of Missed', 'N_unique'], ascending=False, inplace=True)

    if displ:
        display(missed_info)

    if verbose:
        mask = (missed_info['% of Missed'] <= 5.0) & (missed_info['# of Missed'] > 0)

        if mask.any():
            print("Suggested to drop rows in the columns:\n")
            for col in missed_info.index[mask]:
                print(
                    f"Column '{col} with {missed_info.loc[col, '% of Missed']}% \
                    of missed rows ({missed_info.loc[col, '# of Missed']} in total)'")

    return missed_info


# Calculate normality tests and provide plot or varbose
# noinspection PyTypedDict
def tests_for_normality(array, sign_val=0.05, plot=False, verbose=True, desc_stats=True):
    """
    This function provide 3 tests for normality Shapiro-Wilk, NormaTest, Anderson-Darling and provide plot.

    Args:
        array (array-like with shape (n, )): array-like values for testing without NaN
        sign_val (float): significant level for p-values
        plot (bool): True for distribution plot and box-plot
        verbose (bool): True for print suggestions
        desc_stats (bool): True for describe function
    Returns:
        results (dict): dictionary of the results of tests
    """
    if isinstance(array, pd.Series):
        pass
    else:
        try:
            array = pd.Series(array)
        except ValueError:
            print("Check input type.")

    # check for null values
    assert array.notnull().all() is True, "WARNING! The Null values are present!"

    # NormalTest
    _, p_val = normaltest(array)
    results = {"NormalTest": ((True if p_val > sign_val else False), round(p_val, 5))}

    # Shapiro-Wilk
    _, p_val = shapiro(array)
    results["Shapiro_Wilk"] = ((True if p_val > sign_val else False), round(p_val, 5))

    # Anderson-Darling
    st, cr, _ = anderson(array.values)
    results["Anderson_Darling"] = ((True if st < cr[3] else False), None)

    if plot:
        f, axs = plt.subplots(1, 2, figsize=(9, 4))
        sns.distplot(array, kde=True, fit=norm, ax=axs[0])
        axs[0].set_title("Distribution")

        sns.boxplot(array, orient='h', ax=axs[1])
        axs[1].set_title("Box-plot")
        plt.show()

    if desc_stats:
        display(array.describe().to_frame().iloc[1:].T)

    if verbose:
        s_1 = f"The values"
        s_2 = ''
        all_t = True
        all_f = False
        for k, v in results.items():
            all_t = all_t and v[0]
            all_f = all_f or v[0]

            if v[0]:
                s_2 += f' are norm-dist by {k}'
            else:
                s_2 += f' are not norm-dist by {k}'
            if k == 'Anderson_Darling':
                s_2 += ' with sign_val=0.05'
            else:
                s_2 += f' with sign_val={sign_val}'
        if all_t:
            s_2 = f' are norm-dist with all tests and sign_val={sign_val}'
        elif not all_f:
            s_2 = f' are not norm-dist with all tests and sign_val={sign_val}'
        else:
            pass
        print(s_1 + s_2)

        if all_t:
            print('Suggest to use Pearson Correlation because it assumes normality and is sensitive to outliers.')
            results['all'] = (True, None)

        elif not all_f:
            print('Suggest not to use Pearson Correlation because it assumes normality and is sensitive to outliers. \
            Use Spearman Correlation instead.')
            results['all'] = (False, None)
        else:
            results['all'] = {k: v[0] for k, v in results.items()}

    return results


# Plot continuous variable vs target variable
def continuous_vs_categorical_plots(continuous_arr, categorical_arr, desc_stats=True):
    """
    Plot Box-Plot and Violin-Plot for continuous variable vs categorical variable (target)

    Args:
        continuous_arr (array-like with shape (n, )): continuous values
        categorical_arr (array-like with shape (n, )): categorical values (target)
        desc_stats (bool): True for groupby statistics

    Returns:
        Plots and display
    """

    if isinstance(continuous_arr, pd.Series):
        pass
    else:
        try:
            continuous_arr = pd.Series(continuous_arr)
        except ValueError:
            print("Check input type.")

    if isinstance(categorical_arr, pd.Series):
        pass
    else:
        try:
            categorical_arr = pd.Series(categorical_arr)
        except ValueError:
            print("Check input type.")

    a = 1
    n_unique_val = categorical_arr.nunique()

    if n_unique_val > 10:
        print(f"The # of classes is {n_unique_val}. Do you still want to plot? Type 1 or 0.")
        a = int(input())
    else:
        pass

    if a == 1:

        f, axs = plt.subplots(1, 2, figsize=(9, 4))
        sns.violinplot(x=continuous_arr, y=categorical_arr, ax=axs[0], fill=False)
        axs[0].set_title("Violin-plot")

        sns.boxplot(x=continuous_arr, y=categorical_arr, ax=axs[1])
        axs[1].set_title("Box-plot")
        plt.show()

    else:
        print('No plot.')

    if desc_stats:
        df = pd.DataFrame.from_dict({'cont': continuous_arr, 'cat': categorical_arr}, orient="columns")
        display(df.groupby('cat')['cont'].agg(['min', 'mean', 'median', 'max', 'std']))


# Plot categorical variable vs categorical variable
def categorical_vs_categorical_plots(categorical_arr_1, categorical_arr_2, desc_stats=True, verbose=True):
    """
    Plot 100% Stacked Bar-Plot for categorical variable and another categorical variable (target)

    Args:
        categorical_arr_1 (array-like with shape (n, )): categorical values (feature)
        categorical_arr_2 (array-like with shape (n, )): categorical values (target)
        desc_stats (bool): True for groupby statistics
        verbose (bool): True for distribution per category

    Returns:
        Plots and display
    """

    if isinstance(categorical_arr_1, pd.Series):
        pass
    else:
        try:
            categorical_arr_1 = pd.Series(categorical_arr_1)
        except ValueError:
            print("Check input type.")

    if isinstance(categorical_arr_2, pd.Series):
        pass
    else:
        try:
            categorical_arr_2 = pd.Series(categorical_arr_2)
        except ValueError:
            print("Check input type.")

    a = 1
    n_classes = categorical_arr_1.nunique()
    n_unique = categorical_arr_2.nunique()

    if n_classes > 6:
        print(f"The # of classes is {n_classes}. Do you still want to plot? Type 1 or 0.")
        a = int(input())
    if n_unique > 25:
        print(f"The # of unique values is {n_unique}. Do you still want to plot? Type 1 or 0.")
        a = int(input())
    else:
        pass

    if a == 1:

        idx = categorical_arr_1.value_counts().index

        cross_tab_prop = pd.crosstab(index=categorical_arr_1,
                                     columns=categorical_arr_2,
                                     normalize="index")

        cross_tab_prop = cross_tab_prop.loc[idx]

        cross_tab = pd.crosstab(index=categorical_arr_1,
                                columns=categorical_arr_2)

        cross_tab = cross_tab.loc[idx]

        cross_tab_prop.plot(kind='bar',
                            stacked=True,
                            colormap='tab10',
                            figsize=(20, 6))

        plt.legend(loc="lower left", ncol=2)
        plt.xlabel("Release Year")
        plt.ylabel("Proportion")

        for n, x in enumerate([*cross_tab.index.values]):
            for (proportion, count, y_loc) in zip(cross_tab_prop.loc[x],
                                                  cross_tab.loc[x],
                                                  cross_tab_prop.loc[x].cumsum()):
                plt.text(x=n - 0.17,
                         y=(y_loc - proportion) + (proportion / 2),
                         s=f'{count}\n({np.round(proportion * 100, 1)}%)',
                         color="black",
                         fontsize=10,
                         fontweight="bold")

        plt.show()

    else:
        print('No plot.')

    if verbose:
        cross = pd.crosstab(categorical_arr_2, categorical_arr_1)
        s = cross.sum(axis=0)
        cross = ((cross / s) * 100).round(2)
        cross = cross.astype(str)
        cross = cross.apply(lambda x: x + "%")
        cross.loc['total'] = s
        cross.sort_values(by='total', axis=1, ascending=False, inplace=True)
        cross.loc['total %'] = ((s / categorical_arr_1.shape[0]) * 100).round(2)
        cross.loc['total %'] = cross.loc['total %'].astype(str)
        cross.loc['total %'] = cross.loc['total %'].apply(lambda x: x + "%")
        display(cross)

    if desc_stats:
        df = pd.DataFrame.from_dict({'cat_1': categorical_arr_1, 'cat_2': categorical_arr_2}, orient="columns")
        result = df.groupby('cat_2').agg(
            top=('cat_1', lambda x: x.value_counts().idxmax()),
            frequent_t=('cat_1', lambda x: x.value_counts().max()),
            bottom=('cat_1', lambda x: x.value_counts().idxmin()),
            frequent_b=('cat_1', lambda x: x.value_counts().min())
        )
        display(result)
