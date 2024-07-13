import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.plotting.register_matplotlib_converters()


"""Statistics"""
alpha = 0.05  # Significance level
confidence_level = 0.95

cmap='rocket'

def phi_corr_matrix(df: pd.DataFrame, feature_list):
    """Compute and visualize Phi correlation matrix for binary features"""
    corr_matrix = pd.DataFrame(index=feature_list, columns=feature_list)

    for i in range(len(feature_list)):
        for j in range(i, len(feature_list)):
            feature1 = feature_list[i]
            feature2 = feature_list[j]
            corr_coef = matthews_corrcoef(df[feature1], df[feature2])
            corr_matrix.loc[feature1, feature2] = corr_coef
            corr_matrix.loc[feature2, feature1] = corr_coef

    mask = np.triu(np.ones(corr_matrix.shape), k=0).astype(bool)
    filtered_matrix = corr_matrix.mask(mask)

    sns.heatmap(filtered_matrix.astype(float), annot=True, annot_kws={"size": 8},
                cmap=cmap, fmt=".2f", vmin=-1, vmax=1)
    plt.title('Phi Correlation Matrix of Binary Attributes')
    plt.show()



def chi_squared_test(df: pd.DataFrame, feature_tuple):
    """ Chi-squared test for binary features """

    contingency_table = pd.crosstab(df[feature_tuple[0]], df[feature_tuple[1]])

    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    a = f"Chi-squared statistic: {chi2}."
    b = f"P-value: {p_value}."

    if p_value < alpha:
        print(f"{a} {b} Reject the null hypothesis.")
    else:
        print(f"{a} {b} Do not reject the null hypothesis.")


def biserial_heatmap(df: pd.DataFrame, continues_features, binary_features):
    """ Biserial correlation for binary and continues features. """
    correlation_matrix = pd.DataFrame(
        index=binary_features, columns=continues_features)

    for binary_feature in binary_features:

        for continuous_feature in continues_features:

            biserial_corr, _ = stats.pointbiserialr(

                df[binary_feature], df[continuous_feature])

            correlation_matrix.loc[binary_feature,

                                   continuous_feature] = biserial_corr

    correlation_matrix = correlation_matrix.apply(pd.to_numeric)

    sns.heatmap(pd.DataFrame(correlation_matrix),
                cmap=cmap, fmt=".2f")

    plt.title(f"Biserial Correlation Heatmap")

    plt.show()


def pearson_heatmap(df: pd.DataFrame, continuous_features, target_continuous_feature):
    """ Pearson correlation for continuous features against a target continuous feature. """
    correlation_matrix = pd.DataFrame(index=[target_continuous_feature], columns=continuous_features)

    for continuous_feature in continuous_features:
        pearson_corr, _ = stats.pearsonr(df[target_continuous_feature], df[continuous_feature])
        correlation_matrix.loc[target_continuous_feature, continuous_feature] = pearson_corr

    correlation_matrix = correlation_matrix.apply(pd.to_numeric)

    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title(f"Pearson Correlation Heatmap for {target_continuous_feature}")
    plt.show()

    
def confidence_intervals(data: pd.DataFrame, type) -> None:
    """Calculate Confidence Intervals for a given dataset."""

    sample_mean = np.mean(data)

    if type == 'Continuous':
        # ddof=1 for sample standard deviation
        sample_std = np.std(data, ddof=1)
        critical_value = stats.norm.ppf((1 + confidence_level) / 2)

    elif type == 'Discrete':
        # Sample standard deviation for discrete data
        sample_std = np.sqrt(np.sum((data - sample_mean)**2) / (len(data) - 1))
        # t-distribution for discrete data
        critical_value = stats.t.ppf(
            (1 + confidence_level) / 2, df=len(data) - 1)

    standard_error = sample_std / np.sqrt(len(data))
    margin_of_error = critical_value * standard_error

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    print(f"Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")


def significance_t_test(df: pd.DataFrame, feature: str, change_feature: str,
                        min_change_value: float, max_change_value: float) -> None:
    """Perform a t-test (sample size is small or when 
    the population standard deviation is unknown) and follows a normal distribution."""
    t_stat, p_value = stats.ttest_ind(df[df[change_feature] == min_change_value][feature],
                                      df[df[change_feature] == max_change_value][feature], equal_var=False)

    if p_value < alpha:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Reject null hypothesis')
    else:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Fail to reject null hypothesis')



def significance_mannwhitneyu(df: pd.DataFrame, feature: str, change_feature: str,
                        min_change_value: float, max_change_value: float) -> None:
    """
    Perform a Mann-Whitney U, when:
      1. data are ordinal or continuous but not normally distributed
      2. two samples are independent.
      """
    t_stat, p_value = stats.mannwhitneyu(df[df[change_feature] == min_change_value][feature],
                                      df[df[change_feature] == max_change_value][feature])

    if p_value < alpha:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Reject null hypothesis')
    else:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Fail to reject null hypothesis')




def pearson_correlation_test(df: pd.DataFrame, feature, change_feature):

    correlation, p_value = stats.pearsonr(df[feature], df[change_feature])
    
    if p_value < alpha:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Reject null hypothesis')
    else:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Fail to reject null hypothesis')



def vif(df: pd.DataFrame):
    """Calculating Variance Inflation Factor (VIF)."""
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]

    return (vif)