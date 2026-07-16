import matplotlib
matplotlib.use('Agg') # Headless backend for web server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os

# Set custom aesthetics for a premium look
plt.style.use('ggplot')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.facecolor': '#1E1E2E',  # Dark theme background
    'axes.facecolor': '#252538',
    'axes.edgecolor': '#44445c',
    'axes.labelcolor': '#CDD6F4',
    'xtick.color': '#A6ADC8',
    'ytick.color': '#A6ADC8',
    'text.color': '#CDD6F4',
    'figure.titlesize': 14,
    'axes.titlesize': 12
})

def generate_correlation_heatmap(df, output_path):
    """
    Generate Pearson correlation heatmap for numeric columns.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        # Generate an empty informative chart
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Insufficient numeric columns\nfor correlation analysis", 
                ha='center', va='center', color='#F38BA8', fontsize=12)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        return
        
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(max(6, min(12, corr.shape[1])), max(5, min(10, corr.shape[0]))))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)
    
    ticks = np.arange(0, len(corr.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha='left')
    ax.set_yticklabels(corr.columns)
    
    # Add correlation values inside heatmap cells
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", 
                    ha='center', va='center', 
                    color='black' if abs(corr.iloc[i, j]) > 0.4 else '#CDD6F4', 
                    fontsize=8 if len(corr.columns) > 10 else 10)
                    
    plt.title("Pearson Correlation Heatmap", pad=20, color='#CDD6F4')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def generate_confusion_matrix(cm, labels, output_path):
    """
    Generate Confusion Matrix plot for Classification results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax, shrink=0.8)
    
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='left')
    ax.set_yticklabels(labels)
    
    # Set labels
    ax.set_xlabel('Predicted Label', color='#CDD6F4')
    ax.set_ylabel('True Label', color='#CDD6F4')
    
    # Add cell counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i][j]
            ax.text(j, i, f"{val}", 
                    ha='center', va='center', 
                    color='white' if val > (np.max(cm) / 2) else '#CDD6F4')
                    
    plt.title("Classification Confusion Matrix", pad=20, color='#CDD6F4')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def generate_feature_importance_chart(importances, output_path):
    """
    Generate Feature Importance horizontal bar chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Take top 10 features
    top_features = list(importances.keys())[:10]
    top_values = list(importances.values())[:10]
    
    # Reverse so highest is at the top of horizontal bar chart
    top_features.reverse()
    top_values.reverse()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(top_features, top_values, color='#89B4FA') # Light blue neon
    
    # Highlight the top feature with a different color
    if bars:
        bars[-1].set_color('#CBA6F7') # Purple neon
        
    ax.set_xlabel('Relative Importance / Attribution', color='#CDD6F4')
    plt.title("Top Feature Importances", color='#CDD6F4')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def generate_clustering_chart(X, labels, output_path):
    """
    Reduce data dimension with PCA and plot 2D KMeans clustering result.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='none', s=40)
    
    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="best")
    ax.add_artist(legend1)
    
    ax.set_xlabel('PCA 1', color='#CDD6F4')
    ax.set_ylabel('PCA 2', color='#CDD6F4')
    plt.title("KMeans Clustering Profile (PCA Projected)", color='#CDD6F4')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def generate_anomaly_chart(X, predictions, output_path):
    """
    Reduce dimension with PCA and plot normal vs anomalous points.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # predictions: 1 is normal, -1 is anomaly
    normal_mask = (np.array(predictions) == 1)
    anomaly_mask = (np.array(predictions) == -1)
    
    # Scatter normal
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], c='#89B4FA', label='Normal', alpha=0.6, s=30)
    # Scatter anomalies
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], c='#F38BA8', label='Anomaly', alpha=0.9, edgecolors='black', s=50, marker='X')
    
    ax.set_xlabel('PCA 1', color='#CDD6F4')
    ax.set_ylabel('PCA 2', color='#CDD6F4')
    ax.legend(loc="best")
    plt.title("IsolationForest Anomalies (PCA Projected)", color='#CDD6F4')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
