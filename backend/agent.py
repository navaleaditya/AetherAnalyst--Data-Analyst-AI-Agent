import os
import json
import logging
import pandas as pd
import numpy as np
import polars as pl
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Import local modules
from backend.models import preprocess_data, train_classification, train_clustering, train_anomaly_detection
from backend.charts import (
    generate_correlation_heatmap, generate_confusion_matrix, 
    generate_feature_importance_chart, generate_clustering_chart, generate_anomaly_chart
)
from backend.pdf_report import generate_pdf_report

logger = logging.getLogger(__name__)

# State definition
class AgentState(TypedDict):
    session_id: str
    file_path: str
    df_shape: List[int]
    df_columns: List[str]
    df_profile: Dict[str, Any]
    correlation_summary: List[Dict[str, Any]]
    task_type: Optional[str]
    target_column: Optional[str]
    ml_results: Optional[Dict[str, Any]]
    charts: Dict[str, str]
    status: str
    reasoning: Optional[str]
    report_path: Optional[str]
    error: Optional[str]
    logs: List[str]

# Define LLM Task Detection
def detect_ml_task(df: pl.DataFrame, logs: List[str]) -> tuple[str, Optional[str], str]:
    """
    Call Gemini 2.5 Flash to inspect data structure and determine the best ML task.
    If Gemini API key is missing or call fails, fall back to robust rule-based heuristics.
    """
    import google.generativeai as genai
    
    # 1. Build heuristic fallback logic
    columns = df.columns
    target_candidates = ["target", "label", "class", "churn", "clicked", "target_label", "outcome", "status", "species", "type"]
    detected_target = None
    
    for col in columns:
        if col.lower() in target_candidates:
            detected_target = col
            break
            
    is_anomaly = False
    for col in columns:
        if "anomaly" in col.lower() or "outlier" in col.lower() or "fraud" in col.lower():
            is_anomaly = True
            break
            
    heuristic_task = "clustering"
    heuristic_reasoning = "Rule-based Fallback: Unsupervised clustering model selected because no label or target columns were identified."
    
    if detected_target:
        heuristic_task = "classification"
        heuristic_reasoning = f"Rule-based Fallback: Supervised classification model selected because a target/label column '{detected_target}' was identified in the dataset."
    elif is_anomaly or any("anomaly" in str(col).lower() for col in columns):
        heuristic_task = "anomaly"
        heuristic_reasoning = "Rule-based Fallback: Unsupervised anomaly detection selected because potential anomaly/outlier signatures were detected."
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        msg = f"GEMINI_API_KEY not found in environment. Using rule-based fallback: task={heuristic_task}."
        logs.append(msg)
        logger.warning(msg)
        return heuristic_task, detected_target, heuristic_reasoning

    try:
        genai.configure(api_key=api_key)
        # Try gemini-2.5-flash, fall back to gemini-1.5-flash if needed
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
            model = genai.GenerativeModel("gemini-1.5-flash")

        # Prepare dataset metadata for the LLM
        shape = list(df.shape)
        dtypes = {name: str(dtype) for name, dtype in df.schema.items()}
        null_counts = {name: int(df[name].null_count()) for name in df.columns}
        unique_counts = {name: int(df[name].n_unique()) for name in df.columns}
        
        # Get head sample (max 3 rows)
        sample = df.head(3).to_dicts()
        
        # Calculate summary statistics (numeric columns)
        numeric_cols = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
        stats_clean = {}
        if numeric_cols:
            desc = df.select(numeric_cols).describe()
            stats_list = desc["statistic"].to_list()
            for col in numeric_cols:
                col_stats = {}
                for i, stat in enumerate(stats_list):
                    val = desc[col][i]
                    col_stats[stat] = float(val) if val is not None else 0.0
                stats_clean[col] = {
                    "mean": col_stats.get("mean", 0.0),
                    "min": col_stats.get("min", 0.0),
                    "max": col_stats.get("max", 0.0),
                    "std": col_stats.get("std", 0.0)
                }

        prompt = f"""
Analyze the following dataset metadata and determine the most appropriate Machine Learning task.
Available Tasks:
1. classification: Choose this if there is a distinct label/target column that we should predict. Look for categorical or low-cardinality integer columns that look like outcomes, classes, target, label, churn, clicked, success, or status.
2. clustering: Choose this if there is no obvious target column and we want to group similar items together.
3. anomaly: Choose this if we want to detect outliers, fraud, or rare abnormal records.

Dataset Summary:
- Shape: {shape[0]} rows, {shape[1]} columns.
- Columns: {columns}
- Data Types: {dtypes}
- Unique Values per column: {unique_counts}
- Missing Values per column: {null_counts}
- Numerical statistics: {stats_clean}
- Top 3 sample records: {sample}

You MUST reply with a JSON object in this format:
{{
  "task_type": "classification" | "clustering" | "anomaly",
  "target_column": "<name_of_target_column>" | null,
  "reasoning": "<highly detailed explanation of why this task was chosen, including target variable assessment if classification is chosen>"
}}
"""
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        result = json.loads(response.text)
        task_type = result.get("task_type", "clustering")
        target_column = result.get("target_column")
        reasoning = result.get("reasoning", "No explanation provided.")
        
        # Validate task type
        if task_type not in ["classification", "clustering", "anomaly"]:
            task_type = "clustering"
            
        # If target column is specified but not in dataset, reset to None
        if target_column and target_column not in columns:
            target_column = None
            if task_type == "classification":
                task_type = "clustering"
                reasoning += " (Target column specified was invalid, falling back to clustering)"
                
        return task_type, target_column, reasoning
        
    except Exception as e:
        msg = f"Failed to call Gemini API: {e}. Using rule-based fallback: task={heuristic_task}."
        logs.append(msg)
        logger.error(msg)
        return heuristic_task, detected_target, f"{heuristic_reasoning} (API call failed: {e})"




# --- LANGGRAPH NODE FUNCTIONS ---

def profiler_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Executing Dataset Profiler node...")
    
    try:
        file_path = state["file_path"]
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        else:
            try:
                df = pl.read_excel(file_path)
            except Exception:
                import pandas as pd
                df = pl.from_pandas(pd.read_excel(file_path))
        
        # Profile features
        df_shape = list(df.shape)
        df_columns = df.columns
        
        # Profile types and statistics
        data_types = {name: str(dtype) for name, dtype in df.schema.items()}
        missing_values = {name: int(df[name].null_count()) for name in df.columns}
        unique_values = {name: int(df[name].n_unique()) for name in df.columns}
        
        # Build summary stats dictionary
        desc_df = df.describe()
        summary_stats = {}
        stats_list = desc_df["statistic"].to_list()
        for col in df.columns:
            if col != "statistic" and col in desc_df.columns:
                col_stats = {}
                for i, stat in enumerate(stats_list):
                    val = desc_df[col][i]
                    if val is not None:
                        try:
                            val_str = str(val)
                            if '.' in val_str or 'e' in val_str or 'E' in val_str:
                                val = float(val)
                            else:
                                val = int(val)
                        except ValueError:
                            val = str(val)
                    col_stats[stat] = val
                summary_stats[col] = col_stats
                
        df_profile = {
            "num_rows": df_shape[0],
            "num_cols": df_shape[1],
            "missing_values": missing_values,
            "data_types": data_types,
            "unique_values": unique_values,
            "summary_stats": summary_stats
        }
        
        # Convert any nan/inf values to serializable types
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(x) for x in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj
            
        df_profile = make_serializable(df_profile)
        
        logs.append(f"Profiled dataset with Polars: {df_shape[0]} rows, {df_shape[1]} columns.")
        return {
            "df_shape": df_shape,
            "df_columns": df_columns,
            "df_profile": df_profile,
            "status": "correlation_analysis",
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Profiler error: {e}")
        return {
            "error": f"Failed to profile dataset: {str(e)}",
            "status": "error",
            "logs": logs
        }

def correlation_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Executing Correlation Analyzer node...")
    
    try:
        file_path = state["file_path"]
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        else:
            try:
                df = pl.read_excel(file_path)
            except Exception:
                import pandas as pd
                df = pl.from_pandas(pd.read_excel(file_path))
                
        numeric_cols = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
        
        charts = state.get("charts", {})
        heatmap_name = f"heatmap_{state['session_id']}.png"
        heatmap_path = os.path.join("static", "charts", heatmap_name)
        
        # Generate chart using pandas df fallback for matplotlib
        df_pandas = df.to_pandas()
        generate_correlation_heatmap(df_pandas, heatmap_path)
        charts["correlation_heatmap"] = f"/static/charts/{heatmap_name}"
        
        # Extract top correlations and build correlation matrix grid
        correlation_summary = []
        correlation_matrix = {}
        if len(numeric_cols) >= 2:
            corr_grid = {}
            for col1 in numeric_cols:
                corr_grid[col1] = {}
                for col2 in numeric_cols:
                    corr_val = df.select(pl.corr(col1, col2)).item()
                    corr_grid[col1][col2] = corr_val if corr_val is not None and not np.isnan(corr_val) else 0.0
            
            # Find pairs with correlation > 0.5 (excluding diagonal)
            pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    val = corr_grid[col1][col2]
                    pairs.append({"feature1": col1, "feature2": col2, "correlation": float(val)})
            
            # Sort by highest absolute correlation
            correlation_summary = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:5]
            correlation_matrix = {
                "columns": numeric_cols,
                "values": [[corr_grid[c1][c2] for c2 in numeric_cols] for c1 in numeric_cols]
            }
            logs.append(f"Extracted correlation matrix and saved heatmap to {heatmap_path}.")
        else:
            logs.append("Skipped correlation matrix: less than 2 numeric columns.")
            
        return {
            "correlation_summary": correlation_summary,
            "correlation_matrix": correlation_matrix,
            "charts": charts,
            "status": "task_detection",
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Correlation analyzer error: {e}")
        return {
            "error": f"Failed correlation analysis: {str(e)}",
            "status": "error",
            "logs": logs
        }

def task_detector_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Executing ML Task Detector node...")
    
    try:
        file_path = state["file_path"]
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        else:
            try:
                df = pl.read_excel(file_path)
            except Exception:
                import pandas as pd
                df = pl.from_pandas(pd.read_excel(file_path))
                
        task_type, target_col, reasoning = detect_ml_task(df, logs)
        
        logs.append(f"ML Task Detected: {task_type} | Target Variable: {target_col}")
        return {
            "task_type": task_type,
            "target_column": target_col,
            "reasoning": reasoning,
            "status": "model_execution",
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Task detector error: {e}")
        return {
            "error": f"Failed task detection: {str(e)}",
            "status": "error",
            "logs": logs
        }

def model_executor_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Executing Model Executor node...")
    
    try:
        file_path = state["file_path"]
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        else:
            try:
                df = pl.read_excel(file_path)
            except Exception:
                import pandas as pd
                df = pl.from_pandas(pd.read_excel(file_path))
                
        task_type = state["task_type"]
        target_col = state["target_column"]
        
        # Preprocess features and target
        X, y, feature_names = preprocess_data(df, target_column=target_col)
        
        ml_results = None
        if task_type == "classification":
            logs.append("Training Gradient Boosting Classifier...")
            ml_results = train_classification(X, y)
        elif task_type == "clustering":
            logs.append("Training KMeans Clustering model...")
            ml_results = train_clustering(X)
        elif task_type == "anomaly":
            logs.append("Training Isolation Forest Anomaly detector...")
            ml_results = train_anomaly_detection(X)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        # Add 2D PCA projection for interactive scatter plot in the frontend
        if task_type in ["clustering", "anomaly"]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            # Store projection coordinates
            ml_results["pca_projection"] = X_pca.tolist()
            
        logs.append("Model training and evaluation successfully completed.")
        return {
            "ml_results": ml_results,
            "status": "chart_generation",
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Model executor error: {e}")
        return {
            "error": f"Failed model execution: {str(e)}",
            "status": "error",
            "logs": logs
        }

def chart_generator_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Executing Chart Generator node...")
    
    try:
        file_path = state["file_path"]
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        else:
            try:
                df = pl.read_excel(file_path)
            except Exception:
                import pandas as pd
                df = pl.from_pandas(pd.read_excel(file_path))
                
        task_type = state["task_type"]
        target_col = state["target_column"]
        ml_results = state["ml_results"]
        charts = state.get("charts", {})
        
        # Re-preprocess data to get input features matrix for PCA charts
        X, _, _ = preprocess_data(df, target_column=target_col)
        
        model_chart_name = f"model_{state['session_id']}.png"
        model_chart_path = os.path.join("static", "charts", model_chart_name)
        
        if task_type == "classification":
            # Generate feature importance bar chart
            generate_feature_importance_chart(ml_results["feature_importances"], model_chart_path)
            charts["feature_importance"] = f"/static/charts/{model_chart_name}"
            
            # Confusion matrix chart
            cm_chart_name = f"cm_{state['session_id']}.png"
            cm_chart_path = os.path.join("static", "charts", cm_chart_name)
            
            # Extract target classes from mappings
            mapping = ml_results.get("target_mapping") or {}
            labels = [mapping.get(i, str(i)) for i in range(len(mapping))]
            if not labels:
                labels = [str(i) for i in range(len(ml_results["confusion_matrix"]))]
                
            generate_confusion_matrix(ml_results["confusion_matrix"], labels, cm_chart_path)
            charts["confusion_matrix"] = f"/static/charts/{cm_chart_name}"
            
        elif task_type == "clustering":
            # Generate 2D PCA cluster chart
            generate_clustering_chart(X, ml_results["predictions"], model_chart_path)
            charts["cluster_scatter"] = f"/static/charts/{model_chart_name}"
            
        elif task_type == "anomaly":
            # Generate 2D PCA anomaly scatter chart
            generate_anomaly_chart(X, ml_results["predictions"], model_chart_path)
            charts["anomaly_scatter"] = f"/static/charts/{model_chart_name}"
            
        logs.append(f"Visualizations generated successfully and saved to static assets.")
        return {
            "charts": charts,
            "status": "saving_memory",
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Chart generator error: {e}")
        return {
            "error": f"Failed chart generation: {str(e)}",
            "status": "error",
            "logs": logs
        }

def session_memory_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Executing Session Memory node (Report Compilation)...")
    
    try:
        report_name = f"report_{state['session_id']}.pdf"
        report_path = os.path.join("static", "reports", report_name)
        
        # Call helper to generate PDF report
        generate_pdf_report(state, report_path)
        
        # Save complete state to a JSON session memory file
        memory_path = os.path.join("static", "memory", f"{state['session_id']}.json")
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        
        serializable_state = {k: v for k, v in state.items() if k != "logs"}
        # Ensure numpy types are converted
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
                
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, cls=NumpyEncoder, indent=2)
            
        logs.append(f"PDF report compiled and saved to {report_path}.")
        logs.append(f"Session memory stored in {memory_path}.")
        logs.append("Agent execution successfully finished!")
        
        return {
            "report_path": f"/static/reports/{report_name}",
            "status": "completed",
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Session memory error: {e}")
        return {
            "error": f"Failed session compilation: {str(e)}",
            "status": "error",
            "logs": logs
        }


# --- BUILD STATE GRAPH ---

def create_agent_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("dataset_profiler", profiler_node)
    workflow.add_node("correlation_analyzer", correlation_node)
    workflow.add_node("ml_task_detector", task_detector_node)
    workflow.add_node("model_executor", model_executor_node)
    workflow.add_node("chart_generator", chart_generator_node)
    workflow.add_node("session_memory", session_memory_node)
    
    # Add Edges
    workflow.add_edge(START, "dataset_profiler")
    workflow.add_edge("dataset_profiler", "correlation_analyzer")
    workflow.add_edge("correlation_analyzer", "ml_task_detector")
    workflow.add_edge("ml_task_detector", "model_executor")
    workflow.add_edge("model_executor", "chart_generator")
    workflow.add_edge("chart_generator", "session_memory")
    workflow.add_edge("session_memory", END)
    
    # Compile Graph
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer, interrupt_after=["ml_task_detector"])
    return workflow.compile()
