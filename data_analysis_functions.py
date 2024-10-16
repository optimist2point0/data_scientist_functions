import os
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display_functions import display

from scipy.stats import chi2_contingency, shapiro, norm, normaltest, anderson, kstest, probplot

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


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
    Calculate cross-table (confusion table) for Chi2-test (G-test, Monte Carlo simulation).

    Args:
        feature_1: array-like of shape (n, ) with values to calculate cross-table
        feature_2: array-like of shape (n, ) with values to calculate cross-table
    Returns:
        cross_table (pd.DataFrame): cross-table
        lam (int or None): lambda_ parameter determines Chi2-test (1) or G-test (0) or Monte Carlo simulation (-1) or
                           return 1.0 for Cramers' vV correlation (None)
        text (str or None): None or text of WARNING
        yate_correction (bool or None): parameter if to use Yates’ correction for continuity
    """

    # If this is the same arrays
    if np.all(feature_1 == feature_2):
        return pd.crosstab(feature_1, feature_2), None, None, None

    else:
        # calculate cross-table (contingency table)
        cross_table = pd.crosstab(feature_1, feature_2)
        # Chi2 test by default
        lam = 1
        # if any warning
        text = None
        # use of Yates’ correction for continuity
        yate_correction = False
        # degree of freedom
        dff = (cross_table.shape[0] - 1) * (cross_table.shape[1] - 1)
        # calculate expected values
        expected_vals = (
                                cross_table.values.sum(axis=0).reshape(-1, 1)
                                * cross_table.values.sum(axis=1).reshape(1, -1)
                        ) / cross_table.sum().sum()

        # check the rule if to use the Yates’ correction for continuity or not
        if (cross_table < 10).any().any() and dff == 1:
            yate_correction = True
        else:
            pass

        if ((cross_table < 5).sum().sum() / cross_table.size > 0.2) or (cross_table < 1).any().any():
            text = ("WARNING! There are > 20% of cells with frequencies < 5 or at least one cell with 0. "
                    "Monte Carlo simulation will be used  instead of Chi2 (G-test). "
                    "It returns p-value for independency test.")
            return cross_table, -1, text, None

        elif np.any(expected_vals) < 3:
            text = "WARNING! There are small E-xpected values. G-test will be used instead of Chi2."
            lam = 0
            return cross_table, lam, text, yate_correction

        else:
            return cross_table, lam, text, yate_correction


# Monte Carlo simulation for Chi2 test and p-value
def monte_carlo_chi2_test(cross_table, num_simulations=10000):
    """
    Calculate p-value for independency test of Chi2 using Monte Carlo simulation.

    Args:
        cross_table (pd.DataFrame): cross-table
        num_simulations (int): number of simulation
    Returns:
        p_value (float): p-value of the test
    """
    # Calculate expected values and probabilities
    expected_vals = (
                            cross_table.values.sum(axis=1).reshape(-1, 1)*cross_table.values.sum(axis=0).reshape(1, -1)
                    )/cross_table.sum().sum()
    expected_vals = expected_vals.reshape(-1, )

    # check for 0 in expected values
    eps = 1e-5
    m = expected_vals == 0
    if np.any(m):
        eps = eps / m.sum()
        expected_vals[m] += eps
        expected_vals[~m] -= eps

    probabilities = expected_vals/cross_table.sum().sum()
    # Size of the cross-table
    no_cells = cross_table.values.size
    # Total instances
    total_counts = cross_table.sum().sum()

    # Make simulations
    simulations = np.random.choice(np.arange(no_cells),
                                   size=(num_simulations, total_counts),
                                   p=probabilities)

    # Calculate Chi2 statistics
    def calculate_chi_squared(sim, expected, k):
        sim_table = np.bincount(sim, minlength=k)
        chi_squared = np.sum((sim_table - expected) ** 2 / expected)
        return chi_squared

    # Apply the chi-squared function over the simulations
    chi = [calculate_chi_squared(sim, expected_vals, no_cells) for sim in simulations]

    # Calculating the test statistic from the actual sales data
    statistic = np.sum((cross_table.values.reshape(-1,) - expected_vals) ** 2 / expected_vals)

    # Calculating the p-value
    p_value = (1 + np.sum(np.array(chi) >= statistic)) / (num_simulations + 1)

    return p_value


# Cramers' V correlation for cat. col.
def cramers_v(cross_table, lam, text, yate_correction, num_simulations=10000):
    """
    Calculate Cramér's V statistic for categorical-categorical association.

    Args:
        cross_table (pd.DataFrame): cross-table of 2 variables
        lam (int or None): lambda_ parameter determines Chi2-test (1) or G-test (0) or Monte Carlo simulation (-1)
        text (str or None): None or text of WARNING
        yate_correction (bool): correction parameter determines if apply Yates’ correction for continuity
        num_simulations (int): number of simulation for Monte Carlo simulation
    Returns:
        v (float): Cramers' V correlation
    """
    if lam is None:
        return 1.0

    elif lam == -1:
        if text is not None:
            print(text)
        p_value = monte_carlo_chi2_test(cross_table, num_simulations=num_simulations)
        return p_value

    else:
        # Chi2_contingency returns the chi2 value, p-value, degrees of freedom, and expected frequencies
        chi2, _, _, _ = chi2_contingency(cross_table, lambda_=lam, correction=yate_correction)

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

    count_g_test = 0
    count_monte_carlo = 0
    monte_carlo_list = []

    # Iterate over all pairs of columns
    for i in range(n):
        for j in range(i, n):

            # Create a contingency table for the pair of columns
            cross_table, lam, text, yate_correction = confusion_table(df[cat_cols[i]], df[cat_cols[j]])

            if lam == 0:
                count_g_test += 1
                if count_g_test < 4:
                    print(text)

            if lam == -1:
                count_monte_carlo += 1
                if count_monte_carlo > 3:
                    text = None
                else:
                    pass

            # Calculate Cramér's V for this pair
            cramers_v_value = cramers_v(cross_table, lam, text, yate_correction)

            cramers_v_df.iat[i, j] = cramers_v_value
            cramers_v_df.iat[j, i] = cramers_v_value

            if lam == -1:
                monte_carlo_list.append((cat_cols[i], cat_cols[j], cramers_v_value))

    if 0 < len(monte_carlo_list) < 6:
        for gr in monte_carlo_list:
            print(f"Cols {gr[0]} and {gr[1]} p-value is {gr[2]}")
        print("Friendly remainder: if p-value < sign_value, reject the H_0 (there is statistically significant "
              "evidence to suggest an association between the two variables).")

        return cramers_v_df, None
    else:
        # create dictionary for convenience
        monte_carlo_df = pd.DataFrame(monte_carlo_list, columns=['Col1', 'Col2', 'p_value'])

        return cramers_v_df, monte_carlo_df


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
def tests_for_normality(array, sign_val=0.05, plot=True, verbose=False, desc_stats=True):
    """
    This function provide 3 tests for normality Shapiro-Wilk, NormaTest, Anderson-Darling, Kolmogorov-Smirnov
    and provide plots.

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
    assert bool(array.notnull().all()) is True, "WARNING! The Null values are present!"

    # NormalTest
    st, p_val = normaltest(array)
    results = {"NormalTest": ((True if p_val > sign_val else False), round(st, 5), round(p_val, 5))}

    # Shapiro-Wilk
    st, p_val = shapiro(array)
    results["Shapiro_Wilk"] = ((True if p_val > sign_val else False), round(st, 5), round(p_val, 5))

    # Anderson-Darling
    st, cr, _ = anderson(array.values)
    results["Anderson_Darling"] = ((True if st < cr[3] else False), None)

    # Kolmogorov-Smirnov test
    params = norm.fit(array.values)
    st, p_val = kstest(array.values, 'norm', args=params)
    results["Kolmogorov_Smirnov"] = ((True if p_val > sign_val else False), round(st, 5), round(p_val, 5))

    if plot:
        f, axs = plt.subplots(1, 3, figsize=(15, 4))
        sns.distplot(array, kde=True, fit=norm, ax=axs[0])
        axs[0].set_title("Distribution")

        sns.boxplot(array, orient='h', ax=axs[1])
        axs[1].set_title("Box-plot")

        probplot(array.values, plot=axs[2])
        axs[2].set_title("Q-Q Plot")
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


# Return columns sorted in list by its type
def columns_info(df, cat_num_threshold=25):
    """
    Return dictionary of the columns' dtypes as keys and lists of the columns' names as values.

    Args:
         df (pd.DataFrame): DataFrame you worked with
         cat_num_threshold (int): threshold value to determine if the column with int type is numeric for sure or should
                                  be discovered more.

    Returns:
        columns_info_dict (dict): result dictionary with the following keys:
                                    - cat_cols: Columns of categorical type;
                                    - num_cols: Columns of numeric type;
                                    - mix_cols: Columns that don't fit into categorical or numeric;
                                    - miss_cat_cols, miss_num_cols, miss_mix_cols: Columns from each category that
                                                                                   contain missing values.
    """
    columns_info_dict = {
        "cat_cols": [],
        "num_cols": [],
        "mix_cols": [],
        "miss_cat_cols": [],
        "miss_num_cols": [],
        "miss_mix_cols": []
    }

    for col in df.columns:

        if df[col].dtype == 'O':
            columns_info_dict['cat_cols'].append(col)

            if df[col].isnull().any():
                columns_info_dict['miss_cat_cols'].append(col)
            else:
                pass

        elif (df[col].dtype == 'float') or ((df[col].dtype == 'int') and (df[col].nunique() >= cat_num_threshold)):
            columns_info_dict['num_cols'].append(col)

            if df[col].isnull().any():
                columns_info_dict['miss_num_cols'].append(col)
            else:
                pass

        else:
            columns_info_dict['mix_cols'].append(col)

            if df[col].isnull().any():
                columns_info_dict['miss_mix_cols'].append(col)
            else:
                pass

    return columns_info_dict


# Calculate correlation between categorical and continuous values
def categorical_vs_continuous_correlation(categorical_arr, continuous_arr):
    """
    Calculate sqrt of R^2 score (correlation) of an OLS model with X as OneHotEncoded categorical values and y as the
    continuous values.

    Args:
        categorical_arr (array-like with shape (n, 1)): categorical values
        continuous_arr (array-like with shape (n, 1)): continuous values

    Returns:
        correlation_value (float): sqrt of R^2 score (correlation) of OLS model
    """
    assert bool(np.all(pd.notnull(categorical_arr))) is True, ("WARNING! The Null values are present in "
                                                               "categorical_arr!")
    assert np.unique(categorical_arr).shape[0] != 1, "WARNING! One unique value in categorical_arr!"
    assert bool(np.all(pd.notnull(continuous_arr))) is True, ("WARNING! The Null values are present in "
                                                              "continuous_arr!")

    if isinstance(categorical_arr, pd.Series):
        categorical_arr = categorical_arr.to_frame()
    elif isinstance(categorical_arr, np.ndarray):
        categorical_arr = categorical_arr.reshape(-1, 1)

    x = OneHotEncoder(drop='first', sparse_output=False).fit_transform(categorical_arr)

    model = LinearRegression()
    model.fit(x, continuous_arr)

    correlation_value = np.sqrt(model.score(x, continuous_arr))
    return correlation_value


# Calculate overall correlation between features
def correlation_df(df, cat_cols, num_cols, lin_corr_method='pearson', target_col=None, num_simulation=2000):
    """
    Calculate the Cramér's V correlation for all pairs of categorical columns, Pearson or Spearman correlation for all
    pairs of numeric columns and Sqrt of R^2 score for all pairs of numeric and categorical columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame of features
        cat_cols (list): list of categorical columns names, can be empty list
        num_cols (list): list of numeric columns names, can be empty list
        lin_corr_method (string)='pearson': method of linear correlation calculation 'pearson' or 'spearman'
        target_col (string): name for target column (optional)
        num_simulation (int): number of simulations for Monte Carlo
    Returns:
        corr_df (pd.DataFrame): DataFrame of all correlation values (like df.corr())
        monte_carlo_grouped_dict (dict or None): if the Monte Carlo simulations occurs > 5 times, else None
    """
    all_cols = list(num_cols) + list(cat_cols)

    if target_col:
        i = all_cols.index(target_col)
        all_cols.pop(i)
        all_cols.append(target_col)

    n = len(all_cols)
    corr_df = pd.DataFrame(np.zeros((n, n)), index=all_cols, columns=all_cols)

    count_g_test = 0
    count_monte_carlo = 0
    monte_carlo_list = []

    # Iterate over all pairs of columns
    for i in range(n):
        for j in range(i, n):

            if (all_cols[i] in num_cols) and (all_cols[j] in num_cols):
                corr_val = df[[all_cols[i], all_cols[j]]].corr(method=lin_corr_method).iloc[0, 1]

            elif (all_cols[i] in num_cols) and (all_cols[j] in cat_cols):
                corr_val = categorical_vs_continuous_correlation(df[all_cols[j]], df[all_cols[i]])

            elif (all_cols[i] in cat_cols) and (all_cols[j] in num_cols):
                corr_val = categorical_vs_continuous_correlation(df[all_cols[i]], df[all_cols[j]])

            else:
                # Create a contingency table for the pair of columns
                cross_table, lam, text, yate_correction = confusion_table(df[all_cols[i]], df[all_cols[j]])

                if lam == 0:
                    count_g_test += 1
                    if count_g_test < 4:
                        print(text)

                if lam == -1:
                    count_monte_carlo += 1
                    if count_monte_carlo > 3:
                        text = None
                    else:
                        pass

                # Calculate Cramér's V for this pair
                corr_val = cramers_v(cross_table, lam, text, yate_correction, num_simulations=num_simulation)

                if lam == -1:
                    monte_carlo_list.append((all_cols[i], all_cols[j], corr_val))

            corr_df.iat[i, j] = corr_val
            corr_df.iat[j, i] = corr_val

    if count_g_test >= 4:
        print("...")
        print(f"G-test was used {count_g_test} times.")

    if 0 < len(monte_carlo_list) < 6:
        for gr in monte_carlo_list:
            print(f"Cols {gr[0]} and {gr[1]} p-value is {gr[2]}")
        print("Friendly remainder: if p-value < sign_value, reject the H_0 (there is statistically significant "
              "evidence to suggest an association between the two variables).")

        return corr_df, None
    else:
        # create dictionary for convenience
        if monte_carlo_list:
            monte_carlo_df = pd.DataFrame(monte_carlo_list, columns=['Col1', 'Col2', 'p_value'])

            return corr_df, monte_carlo_df

        else:
            return corr_df, None


# Calculate correlation between features and target
def correlation_target_df(df, target_col, cat_cols, num_cols, task='regression', lin_corr_method='pearson',
                          num_simulation=2000):
    """
    Calculate the Cramér's V correlation for categorical columns and target (if it is categorical),
    Pearson or Spearman correlation for numeric columns and target (if it is numeric),
    and Sqrt of R^2 score for numeric&categorical columns and target.

    Args:
        df (pd.DataFrame): DataFrame of features
        target_col (string): name for target column
        cat_cols (list): list of categorical columns names, can be empty list
        num_cols (list): list of numeric columns names, can be empty list
        task (str): how to treat target_col, if 'regression' then numeric; if 'classification' then categorical default
                    'regression'
        lin_corr_method (string)='pearson': method of linear correlation calculation 'pearson' or 'spearman'
        num_simulation (int): number of simulations for Monte Carlo
    Returns:
        corr_df (pd.DataFrame): DataFrame of all correlation values (like df.corr())
        monte_carlo_grouped_dict (dict or None): if the Monte Carlo simulations occurs > 5 times, else None
    """
    assert target_col in df.columns, "WARNING! target_col is not in df."

    all_cols = list(num_cols) + list(cat_cols)

    if target_col in all_cols:
        i = all_cols.index(target_col)
        all_cols.pop(i)
    else:
        pass

    n = len(all_cols)
    corr_df = pd.DataFrame(np.zeros((n, 1)), index=all_cols, columns=[target_col])

    count_g_test = 0
    count_monte_carlo = 0
    monte_carlo_list = []

    # Iterate over all pairs of columns
    for i in range(n):

        if (all_cols[i] in num_cols) and (task == 'regression'):
            corr_val = df[[all_cols[i], target_col]].corr(method=lin_corr_method).iloc[0, 1]

        elif (all_cols[i] in num_cols) and (task == 'classification'):
            corr_val = categorical_vs_continuous_correlation(df[target_col], df[all_cols[i]])

        elif (all_cols[i] in cat_cols) and (task == 'regression'):
            corr_val = categorical_vs_continuous_correlation(df[all_cols[i]], df[target_col])

        else:
            # Create a contingency table for the pair of columns
            cross_table, lam, text, yate_correction = confusion_table(df[all_cols[i]], df[target_col])

            if lam == 0:
                count_g_test += 1
                if count_g_test < 4:
                    print(text)

            if lam == -1:
                count_monte_carlo += 1
                if count_monte_carlo > 3:
                    text = None
                else:
                    pass

            # Calculate Cramér's V for this pair
            corr_val = cramers_v(cross_table, lam, text, yate_correction, num_simulations=num_simulation)

            if lam == -1:
                monte_carlo_list.append((target_col, all_cols[i], corr_val))

        corr_df.iat[i, 0] = corr_val

    if count_g_test >= 4:
        print("...")
        print(f"G-test was used {count_g_test} times.")

    if 0 < len(monte_carlo_list) < 6:
        for gr in monte_carlo_list:
            print(f"Cols {gr[0]} and {gr[1]} p-value is {gr[2]}")
        print("Friendly remainder: if p-value < sign_value, reject the H_0 (there is statistically significant "
              "evidence to suggest an association between the two variables).")

        return corr_df, None

    else:
        # create dictionary for convenience
        if monte_carlo_list:
            monte_carlo_df = pd.DataFrame(monte_carlo_list, columns=['target_col', 'feature', 'p_value'])

            return corr_df, monte_carlo_df

        else:
            return corr_df, None
