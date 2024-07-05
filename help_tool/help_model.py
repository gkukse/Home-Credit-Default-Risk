import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.feature_selection import mutual_info_classif

pd.plotting.register_matplotlib_converters()

import duckdb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_score, roc_auc_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import featuretools as ft

#from help_tool 
#import help_tool, help_visuals, help_stats, help_model
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='featuretools.computational_backends.feature_set_calculator')

pd.set_option('future.no_silent_downcasting', True)

# Setting graph parameters
sns.set_theme(style='whitegrid')
sns.set_palette('husl')


def zero_variance_features(df):
    variances = df.var()
    zero_variances = variances[(variances < 0.1) | variances.isna()].index.tolist()

    return zero_variances


def aggregated_features(df, id):
    df_simplified = df[[f'{id}', 'SK_ID_CURR']].drop_duplicates().reset_index(drop=True)

    # Creating new aggregated features

    es = ft.EntitySet(id='df_data')

    es = es.add_dataframe(dataframe_name='df_simplified',
                        dataframe=df_simplified,
                        index=f'{id}')

    es = es.add_dataframe(dataframe_name='df',
                        dataframe=df.drop(columns=['SK_ID_CURR']).reset_index(drop=True),
                        make_index=True,
                        index='index')

    es = es.add_relationship(parent_dataframe_name='df_simplified',
                            parent_column_name=f'{id}',
                            child_dataframe_name='df',
                            child_column_name=f'{id}')

    df_feature_matrix, feature_defs = ft.dfs(entityset=es,
                                        target_dataframe_name='df_simplified',
                                        agg_primitives=['mean', 'sum', 'count', 'std', 'max', 'min']
                                        )


    df_feature_matrix = df_feature_matrix.reset_index()


    # Dropping Zero variance features
    df_feature_matrix.drop(
        columns= zero_variance_features(df_feature_matrix), 
        inplace = True
        )
    
    return df_feature_matrix

def model_feature_importance_exteranal(df):

    df.dropna(subset='EXT_SOURCE_1', inplace=True)

    y = df['EXT_SOURCE_1']
    X = df.drop(columns=['TARGET', 'EXT_SOURCE_1', 'SK_ID_CURR', 'SK_ID_PREV'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            #('cat', categorical_transformer, categorical_features)
        ])

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': 0,
    }

    # Define the model
    model = LGBMRegressor(**params, n_estimators=100)

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the metrics in a dictionary
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }

    # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

    # Reshape the DataFrame to have a single row
    metrics_df = metrics_df.set_index('Metric').T

    # Plot the heatmap with annotations
    plt.figure(figsize=(6, 2))
    sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap="coolwarm", cbar=False)
    plt.title("Model Evaluation Metrics")
    plt.show()

    # Extract feature importance
    model = pipeline.named_steps['model']
    feature_importances = model.feature_importances_

    # Get the feature names from the preprocessor
    feature_names = numerical_features #+ list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))

    # Create a DataFrame for feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })

    # Sort the DataFrame by importance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    
    return feature_importance


def model_feature_importance_target(df):

    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'EXT_SOURCE_1', 'SK_ID_CURR', 'SK_ID_PREV'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            #('cat', categorical_transformer, categorical_features)
        ])

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': 0,
        'class_weight': class_weight_dict 
    }

    # Define the model
    model = LGBMClassifier(**params, n_estimators=100)

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # # Evaluate the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Store the metrics in a dictionary
    metrics = {
        'Precision': precision,
        'Accuracy': accuracy,
        'Recall': recall,
        'ROC AUC': roc_auc
    }

    # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

    # Reshape the DataFrame to have a single row
    metrics_df = metrics_df.set_index('Metric').T

    # Plot the heatmap with annotations
    plt.figure(figsize=(6, 2))
    sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap="coolwarm", cbar=False)
    plt.title("Model Evaluation Metrics")
    plt.show()

    # Extract feature importance
    model = pipeline.named_steps['model']
    feature_importances = model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance = pd.DataFrame({
        'feature': numerical_features,
        'importance': feature_importances
    })

    # Sort the DataFrame by importance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    return feature_importance


def plot_feature_importance(feature_importance):
    # Plot feature importance
    plt.figure(figsize=(10, 10))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()


def clustering_k_means(df, n_clusters):
    polars_df = pl.from_pandas(df)

    # Convert Polars DataFrame to a numpy array for K-means
    X = polars_df.to_numpy()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the model to the data
    kmeans.fit(X_scaled)

    # Extract cluster labels
    labels = kmeans.labels_

    # Add cluster labels to the Polars DataFrame
    polars_df = polars_df.with_columns(pl.Series(name="cluster", values=labels))

    # print(polars_df)

    column_names = df.columns.to_list()

    # Visualize the clusters (if the data is 2D or can be reduced to 2D)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f'K-means Clustering with {n_clusters} clusters')
    plt.colorbar()
    plt.xlabel(f'{column_names[0]} (scaled)')
    plt.ylabel(f'{column_names[1]} (scaled)')
    plt.show()  

    return polars_df.to_pandas()['cluster']


def clustering_k_means_test(df):
    # Convert Pandas DataFrame to Polars DataFrame
    polars_df = pl.from_pandas(df)

    # Convert Polars DataFrame to a numpy array for K-means
    X = polars_df.to_numpy()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    inertia = []
    silhouette_scores = []
    K = range(2, 10)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # Plot the Elbow
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')

    # Plot the Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')

    plt.show()
