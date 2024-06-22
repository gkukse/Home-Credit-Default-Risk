"""Helper module for EDA notebook to perform 
data cleaning and preprocessing"""


from scipy.stats import chi2_contingency
import os
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, roc_curve)
from sklearn.model_selection import KFold
from unidecode import unidecode
import textblob
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
pd.plotting.register_matplotlib_converters()


"""Statistics"""
alpha = 0.05  # Significance level
confidence_level = 0.95


def csv_download(relative_path: str) -> pd.DataFrame:
    """Download data."""
    absolute_path = os.path.abspath(relative_path)
    df = pd.read_csv(absolute_path, index_col=False, header=0)

    return df


def first_look(df: pd.DataFrame) -> None:
    """Performs initial data set analysis."""
    df_size = df.shape

    df_type = df.dtypes.to_frame().T.rename(index={df.index[0]: 'dtypes'})
    df_null = df.apply(lambda x: x.isna().sum()).to_frame().T.rename(
        index={df.index[0]: 'Null values, Count'})

    # Copy of df_null for Null %
    df_null_proc = round(df_null / df_size[0] * 100, 1)
    df_null_proc = df_null_proc.rename(
        index={df_null.index[0]: 'Null values, %'})

    info_df = pd.concat([df_type, df_null, df_null_proc])

    print(f'Dataset has {df.shape[0]} observations and {df_size[1]} features')
    print(
        f'Columns with all empty values {df.columns[df.isna().all(axis=0)].tolist()}')
    print(f'Dataset has {df.duplicated().sum()} duplicates')

    return info_df.T




def dummy_columns(df, feature_list):
    """ Created a dummy and replaces the old feature with the new dummy """
    df_dummies = pd.get_dummies(df[feature_list])
    df_dummies = df_dummies.astype(int)

    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=feature_list, inplace=True)

    # Drop '_No' features and leave '_Yes'
    # Replace the original column with new dummy
    df = df.drop(columns=[col for col in df.columns if col.endswith('_No')])
    df.columns = [col.replace('_Yes', '') for col in df.columns]
    return df


def phi_corr_matrix(df, feature_list):
    """Compute and visualize Phi correlation matrix for binary features"""
    corr_matrix = pd.DataFrame(index=feature_list, columns=feature_list)

    # Calculate correlation coefficients
    for i in range(len(feature_list)):
        for j in range(i, len(feature_list)):
            feature1 = feature_list[i]
            feature2 = feature_list[j]
            corr_coef = matthews_corrcoef(df[feature1], df[feature2])
            corr_matrix.loc[feature1, feature2] = corr_coef

    # Filter to upper or lower triangular part based on the parameter
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    filtered_matrix = corr_matrix.where(mask)

    # Plot the correlation matrix
    sns.heatmap(filtered_matrix.astype(float), annot=True, annot_kws={"size": 8},
                cmap='rocket', fmt=".2f", vmin=-1, vmax=1)  # Adjust vmin and vmax as needed
    plt.title(f'Phi Correlation Matrix of Binary Attributes')
    plt.show()


def countplot_per_feature(df, feature_list):
    for i, feature_to_exclude in enumerate(feature_list):
        features_subset = [
            feature for feature in feature_list if feature != feature_to_exclude]

        """ Countplot for 5 features """
        fig, axes = plt.subplots(
            1, len(feature_list)-1, figsize=(20, 3))  # Changed the number of columns to 5

        palette = 'rocket'

        for i, feature in enumerate(features_subset):
            sns.countplot(data=df, x=feature, hue=feature_to_exclude,
                          ax=axes[i], palette=palette)
            axes[i].get_legend().remove()
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.suptitle("Binary feature analysis", size=16, y=1.02)
        plt.legend(title=feature_to_exclude,
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


def chi_squared_test(df, feature_tuple):
    """ Chi-squared test for binary features """

    contingency_table = pd.crosstab(df[feature_tuple[0]], df[feature_tuple[1]])

    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    a = f"Chi-squared statistic: {chi2}."
    b = f"P-value: {p_value}."

    if p_value < alpha:
        print(f"{a} {b} Reject the null hypothesis.")
    else:
        print(f"{a} {b} Do not reject the null hypothesis.")


def biserial_heatmap(df, continues_features, binary_features):
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
                annot=True, cmap="rocket", fmt=".2f")

    plt.title("Biserial Correlation Heatmap")

    plt.show()


def confidence_intervals(data, type) -> None:
    """Calculate Confidence Intervals for a given dataset."""

    sample_mean = np.mean(data)

    if type == 'Continuous':
        # Continuous feature
        # ddof=1 for sample standard deviation
        sample_std = np.std(data, ddof=1)
        critical_value = stats.norm.ppf((1 + confidence_level) / 2)

    elif type == 'Discrete':
        # Discrete feature
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


def feature_transpose(df, feature_list):
    """ Transpose a few features into a new dataframe"""
    thresholds = df[feature_list].T
    thresholds.reset_index(inplace=True)
    thresholds.columns = thresholds.iloc[0]
    thresholds.drop(thresholds.index[0], inplace=True)

    return thresholds


def cross_val_thresholds(fold, X, y, thresholds_df, classifiers):
    """ Cross validation with threshold adjustments """
    kf = KFold(n_splits=fold)
    # Initialize lists to store metric scores and confusion matrices
    metric_scores = {metric: {clf_name: [] for clf_name in classifiers.keys(
    )} for metric in ['accuracy', 'precision', 'recall', 'f1']}
    confusion_matrices = {clf_name: np.zeros(
        (2, 2)) for clf_name in classifiers.keys()}

    for train_index, val_index in kf.split(X):
        X_train_i, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_i, y_val = y.iloc[train_index], y.iloc[val_index]

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_i, y_train_i)

            # Threshold update
            # Assuming binary classification
            scores = clf.predict_proba(X_val)[:, 1]
            optimal_threshold = thresholds_df[clf_name].iloc[0]
            y_pred = (scores > optimal_threshold).astype(int)

            # Calculate metrics
            metric_scores['accuracy'][clf_name].append(
                accuracy_score(y_val, y_pred))
            metric_scores['precision'][clf_name].append(
                precision_score(y_val, y_pred))
            metric_scores['recall'][clf_name].append(
                recall_score(y_val, y_pred))
            metric_scores['f1'][clf_name].append(f1_score(y_val, y_pred))

            # Compute confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            confusion_matrices[clf_name] += cm

    # Calculate average scores
    avg_metric_scores = {metric: {clf_name: np.mean(scores) for clf_name, scores in clf_scores.items(
    )} for metric, clf_scores in metric_scores.items()}

    # Average confusion matrices
    avg_confusion_matrices = {
        clf_name: matrix / fold for clf_name, matrix in confusion_matrices.items()}

    cv_results = []
    for clf_name, scores in avg_metric_scores['accuracy'].items():
        cv_results.append({
            'Classifier': classifiers[clf_name].__class__.__name__,
            'CV Mean Accuracy': np.mean(scores),
            'CV Mean Precision': np.mean(avg_metric_scores['precision'][clf_name]),
            'CV Mean Recall': np.mean(avg_metric_scores['recall'][clf_name]),
            'CV Mean F1': np.mean(avg_metric_scores['f1'][clf_name]),
            'Confusion Matrix': avg_confusion_matrices[clf_name]
        })

    model_info = pd.DataFrame(cv_results)
    return model_info


def cross_validation_param(model_info):
    """ Parameter heatmap """
    heatmap_data = model_info

    heatmap_data.set_index('Classifier', inplace=True)

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5)
    plt.title('Model Performance Metrics')
    plt.show()


def cross_validation_confusion_matrix(model_info):
    """ Cross Validation Matrix """
    f, ax = plt.subplots(2, 5, figsize=(15, 6))
    ax = ax.flatten()

    for i, row in model_info.iterrows():
        cm = row['Confusion Matrix']
        sns.heatmap(cm, ax=ax[i], annot=True, fmt='2.0f')
        ax[i].set_title(f"Matrix for {row['Classifier']}")
        ax[i].set_xlabel('Predicted Label')
        ax[i].set_ylabel('True Label')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


def predict_proba_available(model, X_validation):
    """ Check if predict_proba is available. """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_validation)[:, 1] 
    else:
        y_proba = model.predict(X_validation) 

    return y_proba


def model_confusion_matrix(model, X_validation, y_validation, y_pred):
    conf_matrix = confusion_matrix(y_validation, model.predict(X_validation))


    accuracy = accuracy_score(y_validation, y_pred)
    precision = precision_score(y_validation, y_pred)
    recall = recall_score(y_validation, y_pred)


    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')

    sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['0', '1'], 
            yticklabels=['0', '1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Voting Classifier')
    plt.show()


def encode_weekday(day_of_week):
    """Encode weekday using sine and cosine functions."""
    weekdays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
    # Get the index of the weekday
    index = weekdays.index(day_of_week.upper())
    # Convert index to radians
    radians = (index / 7.0) * 2 * np.pi
    # Encode using sine and cosine
    sin_encoding = np.sin(radians)
    cos_encoding = np.cos(radians)
    return sin_encoding, cos_encoding

# # Assuming df is your DataFrame with the column 'WEEKDAY_APPR_PROCESS_START'
# # Encode weekday for the 'WEEKDAY_APPR_PROCESS_START' column
# sin_encoding, cos_encoding = zip(*application_train['WEEKDAY_APPR_PROCESS_START'].apply(encode_weekday))

# # Add encoded features to the DataFrame
# application_train['WEEKDAY_APPR_PROCESS_START_sin'] = sin_encoding
# application_train['WEEKDAY_APPR_PROCESS_START_cos'] = cos_encoding

# # Display the modified DataFrame
# print(application_train.head())


def encode_weekday_sin(day_of_week):
    """Encode weekday using sine function."""
    weekdays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
    # Get the index of the weekday
    index = weekdays.index(day_of_week.upper())
    # Convert index to radians
    radians = (index / 7.0) * 2 * np.pi
    # Encode using sine function
    sin_encoding = np.sin(radians)
    return sin_encoding
