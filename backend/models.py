import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, silhouette_score
)
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df, target_column=None):
    """
    Preprocess raw dataset: impute missing values, scale numerical columns,
    and encode categorical columns.
    """
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    df_processed = df.copy()
    
    # Identify target and features
    if target_column and target_column in df_processed.columns:
        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])
    else:
        y = None
        X = df_processed

    # Drop columns that are completely null or have no variance (single unique value)
    cols_to_drop = []
    for col in X.columns:
        if X[col].isnull().all() or X[col].nunique() <= 1:
            cols_to_drop.append(col)
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    # Separate numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 1. Impute numerical columns with median
    for col in num_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)
            
    # 2. Impute categorical columns with mode
    for col in cat_cols:
        if X[col].isnull().any():
            mode_series = X[col].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else 'Unknown'
            X[col] = X[col].fillna(mode_val)
            
    # 3. Scaling numerical columns
    scaler = StandardScaler()
    if num_cols:
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
    # 4. Encoding categorical columns using pd.get_dummies
    if cat_cols:
        # Save categorical column mappings for output analysis
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    # Ensure all columns are float/int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
            
    return X, y, X.columns.tolist()

def train_classification(X, y):
    """
    Train a GradientBoostingClassifier on the preprocessed dataset.
    """
    # Force y to be categorical code if it's text
    y_encoded = y
    y_mapping = None
    if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype):
        y_categorical = pd.Categorical(y)
        y_encoded = y_categorical.codes
        y_mapping = {int(code): str(category) for code, category in enumerate(y_categorical.categories)}
    else:
        # If integer but not continuous, get mapping as string
        unique_vals = np.unique(y)
        y_mapping = {int(v): str(v) for v in unique_vals}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
    )
    
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importances = {}
    explanation_method = "native"
    
    try:
        import shap
        # shap might raise exception if TreeExplainer doesn't support the configuration
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # In multiclass, shap_values is a list of arrays. In binary, it might be a single array or list.
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
            
        for name, val in zip(X.columns, mean_shap):
            feature_importances[name] = float(val)
        explanation_method = "shap"
    except Exception as e:
        logger.warning(f"SHAP explanation failed, falling back to native feature importances: {e}")
        importances = model.feature_importances_
        for name, val in zip(X.columns, importances):
            feature_importances[name] = float(val)
            
    # Sort feature importances descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # Convert confusion matrix to nested list for JSON serialization
    cm_list = cm.tolist()
    
    return {
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        },
        "confusion_matrix": cm_list,
        "feature_importances": feature_importances,
        "explanation_method": explanation_method,
        "target_mapping": y_mapping,
        "predictions": y_pred.tolist(),
        "y_test": y_test.tolist()
    }

def train_clustering(X):
    """
    Find best KMeans clustering (k between 2 and 5) based on silhouette score.
    """
    best_k = 3
    best_score = -1.0
    best_labels = None
    scores = {}
    
    # Silhoutte score requires at least 2 clusters and less than samples count
    max_k = min(5, len(X) - 1)
    if max_k >= 2:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            scores[k] = float(score)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
    else:
        # Fallback for very small datasets
        best_k = 2
        km = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        best_labels = km.fit_predict(X)
        best_score = 0.0
        scores[2] = 0.0

    # Retrain/extract best model details
    km = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    km.fit(X)
    centroids = km.cluster_centers_
    
    # Calculate size of each cluster
    unique_labels, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    
    return {
        "metrics": {
            "best_k": int(best_k),
            "silhouette_score": float(best_score),
            "scores_per_k": scores
        },
        "cluster_sizes": cluster_sizes,
        "centroids": centroids.tolist(),
        "predictions": best_labels.tolist(),
        "feature_names": X.columns.tolist()
    }

def train_anomaly_detection(X):
    """
    Train an IsolationForest model for anomaly / outlier detection.
    """
    contamination = 0.05 # Default 5% anomaly rate
    model = IsolationForest(contamination=contamination, random_state=42)
    
    # Predictions: 1 for normal, -1 for anomaly
    predictions = model.fit_predict(X)
    scores = model.decision_function(X) # Higher score -> more normal
    
    # Anomaly rate computation
    anomalies_count = int(np.sum(predictions == -1))
    total_count = len(predictions)
    anomaly_rate = anomalies_count / total_count
    
    return {
        "metrics": {
            "anomaly_rate": float(anomaly_rate),
            "anomalies_count": anomalies_count,
            "total_records": total_count,
            "mean_anomaly_score": float(np.mean(scores))
        },
        "predictions": predictions.tolist(),
        "scores": scores.tolist(),
        "feature_names": X.columns.tolist()
    }
