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


def distribution_check(df: pd.DataFrame) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    df = df.reset_index(drop=True)

    for feature in df.columns:

        if df[feature].dtype.name in ['object', 'bool']:
            pass

        else:

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))

            print(f'{feature}')

            # Outlier check (Box plot)
            df.boxplot(column=feature, ax=axes[0])
            axes[0].set_title(
                f'{feature} ranges from {df[feature].min()} to {df[feature].max()}')

            # Distribution check (Histogram).
            sns.histplot(data=df, x=feature, kde=True, bins=20, ax=axes[1])
            axes[1].set_title(f'Distribution of {feature}')

            # Normality check (QQ plot).
            sm.qqplot(df[feature].dropna(), line='s', ax=axes[2])
            axes[2].set_title(f'Q-Q plot of {feature}')

            plt.tight_layout()
            plt.show()


def heatmap(df: pd.DataFrame, name: str, method: str) -> None:
    """ Plotting the heatmap of correlation matrix """
    plt.figure(figsize=(8, 5))
    corr_matrix = df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                vmin=-1, vmax=1, mask=mask)
    plt.title(f'Correlation {name.capitalize()} Attributes')
    plt.show()


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



""" Encoding Various values """

def convert_flags(df):
    for col in df.columns:
        if col.startswith('FLAG'):
            df[col] = df[col].replace({'Y': 1, 'N': 0}).astype(int)
            df[col] = df[col].infer_objects(copy=False)
    return df

def accompanied(df, target_feature):
    #target_feature = 'NAME_TYPE_SUITE'
    df[target_feature] = np.where(df[target_feature] == 'Unaccompanied', 1, 0)
    return df[target_feature]

def top_five_categories(df, target_feature, n_top):

    df[target_feature] = df[target_feature].replace({'XNA': np.nan})

    top_goods_list = df[target_feature].value_counts().nlargest(n_top).index.to_list()

    df.loc[~df[target_feature].isin(top_goods_list), target_feature] = 'Other'
    return df[target_feature]

def risk_category(df, target_feature):
    df['new_feature'] = df[target_feature].apply(lambda x: 1 if 'low_normal' in x.lower() else 1.5 if 'low_action' in x.lower() else 2 if 'middle' in x.lower() else 3 if 'high' in x.lower() else np.nan)

    return df['new_feature']

def client_type_encoding(df, target_feature):
    #target_feature = 'NAME_CLIENT_TYPE'
    df.loc[df[target_feature] == 'New', target_feature] = 1
    df.loc[df[target_feature] == 'Refreshed', target_feature] = 1.5
    df.loc[df[target_feature] == 'Repeater', target_feature] = 2
    df.loc[df[target_feature] == 'XNA', target_feature] = np.nan 

    df[target_feature] = df[target_feature].astype(float)

    return df[target_feature]


def product_combination(df):

    target_feature = 'PRODUCT_COMBINATION'

    df_new=pd.DataFrame()
    df_new[['CONTRACT_TYPE', 'PRODUCT_PLACE', '0']] = df[target_feature].str.split(' ', n=2, expand=True)

    df_new['PRODUCT_PLACE'] = df_new['PRODUCT_PLACE'].str.rstrip(':').str.rstrip('s').str.lower()

    df_new['PRODUCT_RANK'] = df_new['0'].apply(lambda x: 
        1 if pd.notna(x) and 'low' in x.lower() else 
        2 if pd.notna(x) and 'middle' in x.lower() else 
        3 if pd.notna(x) and 'high' in x.lower() else 
        np.nan
    )

    df_new['PRODUCT_INTEREST'] = df_new['0'].apply(lambda x: 
        1 if pd.notna(x) and 'with interest' in x.lower() else 
        0 if pd.notna(x) and 'without interest' in x.lower() else 
        np.nan
    )

    df[['PRODUCT_PLACE', 'PRODUCT_RANK', 'PRODUCT_INTEREST']] = df_new[['PRODUCT_PLACE', 'PRODUCT_RANK', 'PRODUCT_INTEREST']]
    df.drop(columns=target_feature, inplace=True)
    
    return df

def contract_status(df, target_feature):
    df.loc[df[target_feature] == 'Approved', target_feature] = 2
    df.loc[df[target_feature] == 'Unused offer', target_feature] = 1
    df.loc[df[target_feature] == 'Refused', target_feature] = -1
    df.loc[df[target_feature] == 'Canceled', target_feature] = -2

    df[target_feature] = df[target_feature].astype(float)

    return df[target_feature]

def payment_type(df, target_feature):
    #target_feature = 'NAME_PAYMENT_TYPE'

    df.loc[df[target_feature] == 'Cash through the bank', 'new_feature'] = 1
    df.loc[~df[target_feature].isin(['Cash through the bank', 'XNA']), 'new_feature'] = 0
    df.drop(columns=[target_feature], inplace=True)

    return df['new_feature']

def encode_weekday_sin(day_of_week):
    """Encode weekday using sine function."""
    weekdays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
    index = weekdays.index(day_of_week.upper())
    radians = (index / 7.0) * 2 * np.pi
    sin_encoding = np.sin(radians)
    return sin_encoding

def weekday_encoding(df):
    df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].apply(encode_weekday_sin)
    return df


def cardinality_test(df):
    cardinality_threshold = 50
    hight_cardinality = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() > cardinality_threshold]

    return hight_cardinality




def encode_categories(df):
    """ Encoding various categories"""

    df = convert_flags(df)

    # Binary and/or unknown categories
    binary_replacements = {
        'CODE_GENDER': {'F': 1, 'M': 0, 'XNA': np.nan},
        'NAME_CONTRACT_TYPE': {'Cash loans': 1, 'Revolving loans': 0},
        'EMERGENCYSTATE_MODE': {'Yes': 1, 'No': 0}
    }
    for col, mapping in binary_replacements.items():
        df[col] = df[col].replace(mapping).astype(float)

    df['CODE_GENDER_F'] = df.pop('CODE_GENDER')
    df['NAME_CONTRACT_TYPE_CASH_LOANS'] = df.pop('NAME_CONTRACT_TYPE')

    # One-hot encodings
    one_hot_conditions = {
        'HOUSING_TYPE_House': df['NAME_HOUSING_TYPE'] == 'House / apartment',
        'FONDKAPREMONT_reg_oper_account': df['FONDKAPREMONT_MODE'] == 'reg oper account',
        'HOUSETYPE_flats': df['HOUSETYPE_MODE'] == 'block of flats'
    }
    for new_col, condition in one_hot_conditions.items():
        df[new_col] = condition.astype(int)

    df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].map({
        'Panel': 1,
        'Stone, brick': 2
    }).fillna(3).astype(float)

    education_mapping = {
        'Lower secondary': 1, 
        'Secondary / secondary special': 2,
        'Incomplete higher': 3,
        'Higher education': 4,
        'Academic degree': 5
    }
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(education_mapping).astype(int)

    df['NAME_TYPE_SUITE'] = accompanied(df, 'NAME_TYPE_SUITE')
    df = weekday_encoding(df)

    family_status_mapping = {
        'Unknown': np.nan, 
        'Single / not married': 1, 
        'Separated': 1, 
        'Widow': 1, 
        'Married': 2,
        'Civil marriage': 2
    }
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].map(family_status_mapping).astype(float)
    
    income_type_mapping = {
        'Businessman': 2,
        'Commercial associate': 2,
        'State servant': 2,
        'Working': 1,
        'Student': 0.5,
        'Maternity leave': 0, 
        'Unemployed': 0,
        'Pensioner': 0
    }
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].map(income_type_mapping).astype(float)

    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].apply(lambda x: 
        1 if x == 'Business Entity Type 3' else 
        2 if x == 'Self-employed' else 
        np.nan if x == 'XNA' else 
        3
    ).astype(float)

    occupation_mapping = {
        'Managers': 1,
        'High skill tech staff': 1,
        'Core staff': 1,
        'Laborers': 2,
        'Drivers': 2,
        'Sales staff': 2,
        'Cooking staff': 3,
        'Cleaning staff': 3,
        'Security staff': 3,
        'Low-skill laborers': 3,
        'Medicine staff': 3,
        'Private service staff': 3,
        'Realty agents': 3
    }
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].map(occupation_mapping).astype(float)

    df.drop(columns=[
        'NAME_HOUSING_TYPE', 
        'FONDKAPREMONT_MODE', 
        'HOUSETYPE_MODE'
    ], inplace=True)

    return df


def zero_variance_features(df):
    variances = df.var()
    zero_variances = variances[(variances == 0) | variances.isna()].index.tolist()

    return zero_variances


def categorize_card_activity(status):
    if status in ['Active', 'Signed', 'Returned to the store', 'Demand']:
        return 1
    elif status in ['Approved']:
        return 0
    elif status in ['Completed', 'Amortized debt', 'Canceled']:
        return 2
    else:
        return np.nan


def categorize_risk_profile(status):
    if status in ['Active', 'Signed', 'Approved', 'Completed']:
        return 1
    elif status in ['Amortized debt', 'Canceled']:
        return 2
    elif status in ['Demand', 'Returned to the store']:
        return 3
    else:
        return np.nan
    
def yield_group(status):
    if status in ['low_normal']:
        return 1
    elif status in ['low_action']:
        return 1.5
    elif status in ['middle']:
        return 2
    elif status in ['high']:
        return 3
    else:
        return np.nan
    
def product_type(status):
    if status in ['x-sell']:
        return 1
    elif status in ['walk-in']:
        return 2
    else:
        return np.nan
    
def reject_reason(status):
    if status in ['XAP']:
        return -1
    elif status in ['HC']:
        return 1
    elif status in ['XNA']:
        return np.nan
    else:
        return 0

def product_place(status):
    if status in ['industry', 'other']:
        return 'other'
    else:
        return status