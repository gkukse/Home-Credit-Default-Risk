import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import featuretools as ft
import polars as pl

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, mean_absolute_error, 
                             mean_squared_error, r2_score)


from lightgbm import LGBMClassifier, LGBMRegressor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='featuretools.computational_backends.feature_set_calculator')

# Pandas options
pd.set_option('future.no_silent_downcasting', True)

# Seaborn settings
sns.set_theme(style='whitegrid')
sns.set_palette('husl')


def zero_variance_features(df: pd.DataFrame):
    """ Returns list of low variance features"""
    variances = df.var()
    zero_variances = variances[(variances < 0.1) | variances.isna()].index.tolist()

    return zero_variances


def aggregated_features(df: pd.DataFrame, id: str):
    """ Aggregates and returns feature derivatives """

    if id != 'SK_ID_CURR':
        try:
            df_simplified = df[[f'{id}', 'SK_ID_CURR']].drop_duplicates().reset_index(drop=True)
            df = df.drop(columns=['SK_ID_CURR'])
        except:
            df_simplified = df[[f'{id}']].drop_duplicates().reset_index(drop=True)
    else:
        df_simplified = df[[f'{id}']].drop_duplicates().reset_index(drop=True)


    es = ft.EntitySet(id='df_data')

    es = es.add_dataframe(dataframe_name='df_simplified',
                        dataframe=df_simplified,
                        index=f'{id}')

    es = es.add_dataframe(dataframe_name='df',
                        dataframe=df.reset_index(drop=True),
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
    
    return df_feature_matrix



def model_feature_importance_exteranal(df: pd.DataFrame, target: str):
    """ Feature importance from External Source """

    try: 
        df_filtered = df.dropna(subset=target)
    except:
        df_filtered = df

    try:
        df_filtered = df_filtered.drop(columns='SK_ID_PREV')
    except:
        pass



    y = df_filtered[target]
    X = df_filtered.drop(columns=['TARGET', target, 'SK_ID_CURR'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
        ])

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': 0,
    }

    model = LGBMRegressor(**params, n_estimators=100)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

    metrics_df = metrics_df.set_index('Metric').T

    # Plot the heatmap with annotations
    plt.figure(figsize=(6, 2))
    sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap="coolwarm", cbar=False)
    plt.title("Model Evaluation Metrics")
    plt.show()

    # Extract feature importance
    model = pipeline.named_steps['model']
    feature_importances = model.feature_importances_

    feature_names = numerical_features 

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })

    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    
    return feature_importance


def model_feature_importance_target(df: pd.DataFrame):
    """ Feature importance for TARGET """

    df = df.dropna(subset='TARGET')

    try: 
        df = df.drop(columns=['SK_ID_PREV'])
    except: pass

    try: 
        df = df.drop(columns=['SK_ID_BUREAU'])
    except: pass


    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])


    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': 0,
        'class_weight': 'balanced' 
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

    try:
        feature_names = numerical_features + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
    except: 
        feature_names = numerical_features 

    # Create a DataFrame for feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })

    # Sort the DataFrame by importance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    
    return feature_importance


def plot_feature_importance(feature_importance):
    """ Plot feature importance """
    plt.figure(figsize=(10, 10))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()


def clustering_k_means(df: pd.DataFrame, n_clusters: int):
    """ CLusters with K means """
    polars_df = pl.from_pandas(df)
    X = polars_df.to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    polars_df = polars_df.with_columns(pl.Series(name="cluster", values=labels))

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


def clustering_k_means_test(df: pd.DataFrame):
    """ Ploting Cluster numbers vs Inertia """
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




def predict_proba_available(model, X_validation):
    """ Check if predict_proba is available. """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_validation)[:, 1] 
    else:
        y_proba = model.predict(X_validation) 

    return y_proba



def cross_val_thresholds(fold, X, y, thresholds, classifiers):
    """ Perform cross-validation with threshold adjustments and feature scaling """

    kf = KFold(n_splits=fold, shuffle=True, random_state=42)

    # Initialize dictionaries to store metric scores and confusion matrices
    metric_scores = {metric: {clf_name: [] for clf_name in classifiers}
                     for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    confusion_matrices = {clf_name: np.zeros((2, 2)) for clf_name in classifiers}

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize and fit StandardScaler on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

    for clf_name, clf_pipeline in classifiers.items():
        clf = clf_pipeline

        # Fit classifier on scaled training data
        clf.fit(X_train_scaled, y_train)

        # Get predicted probabilities and optimal threshold
        scores = predict_proba_available(clf, X_val_scaled)
        optimal_threshold = thresholds[clf_name]
        y_pred = (scores > optimal_threshold).astype(int)

        # Calculate metrics
        metric_scores['accuracy'][clf_name].append(accuracy_score(y_val, y_pred))
        metric_scores['precision'][clf_name].append(precision_score(y_val, y_pred))
        metric_scores['recall'][clf_name].append(recall_score(y_val, y_pred))
        metric_scores['f1'][clf_name].append(f1_score(y_val, y_pred))
        metric_scores['auc'][clf_name].append(roc_auc_score(y_val, y_pred))

        # Compute confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        confusion_matrices[clf_name] += cm

    # Calculate average scores and confusion matrices
    avg_metric_scores = {metric: {clf_name: np.mean(scores) for clf_name, scores in clf_scores.items()}
                         for metric, clf_scores in metric_scores.items()}
    avg_confusion_matrices = {clf_name: matrix / fold for clf_name, matrix in confusion_matrices.items()}

    # Prepare CV results as a list of dictionaries
    cv_results = []
    for clf_name  in classifiers:
        cv_results.append({
            'Classifier': clf_name,
            'CV Mean Accuracy': np.mean(metric_scores['accuracy'][clf_name]),
            'CV Mean Precision': np.mean(avg_metric_scores['precision'][clf_name]),
            'CV Mean Recall': np.mean(avg_metric_scores['recall'][clf_name]),
            'CV Mean F1': np.mean(avg_metric_scores['f1'][clf_name]),
            'CV Mean ROC AUC': np.mean(avg_metric_scores['auc'][clf_name]),
            'Confusion Matrix': avg_confusion_matrices[clf_name]
        })

    # Convert results to DataFrame
    model_info = pd.DataFrame(cv_results)
    return model_info



def pipeline_creation(params, X_train, y_train, X_validation, y_validation):
    """ Creates pipeline and print F1 score """
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LGBMClassifier(**params, verbose=-1))
    ])

    # Fit the pipeline on the training data
    my_pype = pipeline.fit(X_train, y_train)

    # Predict on training and validation data
    y_pred_train = pipeline.predict(X_train)
    y_pred_validation = pipeline.predict(X_validation)

    print(f'Precision Score in Training: {precision_score(y_train, y_pred_train):.4f}')
    print(f'Precision Score in Validation: {precision_score(y_validation, y_pred_validation):.4f}')

    return my_pype


def find_threshold(my_pipeline, X_validation, y_validation, name: str):

    # Create thresholds for decision threshold tuning
    thresholds = np.linspace(0, 1, 100)

    # Initialize variables to track best accuracy and threshold
    best_precision = 0
    optimal_threshold = 0

    y_proba = predict_proba_available(my_pipeline, X_validation)

    # Find optimal threshold based on accuracy
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        precision = f1_score(y_validation, y_pred)

        if precision > best_precision:
            best_precision = precision
            optimal_threshold = threshold

    # Use the optimal threshold to predict final labels
    y_pred_optimal = (y_proba > optimal_threshold).astype(int)


    results = []
    results.append({'Model': name,
                    'Optimal_Threshold': optimal_threshold,
                    'Accuracy': accuracy_score(y_validation, y_pred_optimal),
                    'Precision': precision_score(y_validation, y_pred_optimal),
                    'Recall': recall_score(y_validation, y_pred_optimal),
                    'F1_Score': f1_score(y_validation, y_pred_optimal),
                    'AUC': roc_auc_score(y_validation, y_pred_optimal)
                    })

    model_threshol_search = pd.DataFrame(results)

    plt.figure(figsize=(8, 3))
    sns.heatmap(model_threshol_search.set_index(
        'Model'), annot=True, fmt=".2f")
    plt.title('Model Performance Metrics')
    plt.show()

    return optimal_threshold


def model_score_test(models, model_names, thresholds_df, X, y):
    """ Calculate various scores for multiple models"""

    data = []
    for model, label in zip(models, model_names):
        if hasattr(model, 'decision_function'):
            predictions = model.decision_function(X)
        else:
            predictions = model.predict_proba(X)[:, 1]

        optimal_threshold = thresholds_df.set_index('Model').loc[label, 'Optimal_Threshold']
        adjusted_predictions = (predictions > optimal_threshold).astype(int)

        f1 = f1_score(y, adjusted_predictions)
        accuracy = accuracy_score(y, adjusted_predictions)
        precision = precision_score(y, adjusted_predictions)
        recall = recall_score(y, adjusted_predictions)
        auc = roc_auc_score(y, predictions)

        data.append([label, accuracy, precision, recall, f1, auc])

    columns = ["Classifier", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    return pd.DataFrame(data, columns=columns)




def model_evaluations_threshold(model, threshold, X_eval, y_eval):
    """ Returns model evaluation parameters. """

    predictions = model.predict_proba(X_eval)[:, 1]
    adjusted_predictions = (predictions > threshold).astype(int)
    y_predprob = adjusted_predictions

    data = []
    f1 = f1_score(y_eval, y_predprob)
    accuracy = accuracy_score(y_eval, y_predprob)
    precision = precision_score(y_eval, y_predprob)
    recall = recall_score(y_eval, y_predprob)
    auc = roc_auc_score(y_eval, y_predprob)

    name = model.named_steps['classifier'].__class__.__name__

    data.append([name, threshold, accuracy, precision, recall, f1, auc])

    columns = ['Classifier', 'Threshold', "Accuracy", "Precision", "Recall", "F1", "AUC"]

    return pd.DataFrame(data, columns=columns)


def model_evaluations(model, X_eval, y_eval):
    """ Returns model evaluation parameters. """

    y_pred = model.predict(X_eval)
    y_predprob = model.predict_proba(X_eval)[:, 1]

    data = []
    f1 = f1_score(y_eval, y_pred)
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_predprob)

    name = model.named_steps['classifier'].__class__.__name__

    data.append([name, False, accuracy, precision, recall, f1, auc])

    columns = ['Classifier', 'Threshold', "Accuracy", "Precision", "Recall", "F1", "AUC"]

    return pd.DataFrame(data, columns=columns)


def multi_evaluation(model, X_train, y_train, X_validation, y_validation):
    """ 
    Evaluate the model on both training and validation sets and combine the results.
    """

    df_1 = model_evaluations(model, X_train, y_train)
    df_1['Type'] = 'Train'
    df_2 = model_evaluations(model, X_validation, y_validation)
    df_2['Type'] = 'Validate'

    return pd.concat([df_1, df_2], ignore_index=True)


