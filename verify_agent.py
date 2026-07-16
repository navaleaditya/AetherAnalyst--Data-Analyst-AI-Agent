import os
import shutil
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs

# Setup Python path to include current workspace root
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agent import create_agent_graph

def generate_test_data():
    """
    Generate synthetic datasets to verify classification, clustering, and anomaly detection.
    """
    os.makedirs("static/test_datasets", exist_ok=True)
    
    # 1. Classification Dataset
    X_clf, y_clf = make_classification(n_samples=200, n_features=6, n_classes=2, random_state=42)
    cols_clf = [f"feature_{i}" for i in range(6)]
    df_clf = pd.DataFrame(X_clf, columns=cols_clf)
    df_clf["target_label"] = y_clf
    clf_path = "static/test_datasets/test_classification.csv"
    df_clf.to_csv(clf_path, index=False)
    print(f"Generated classification test data at {clf_path}")

    # 2. Clustering Dataset
    X_cls, _ = make_blobs(n_samples=200, n_features=4, centers=3, random_state=42)
    cols_cls = [f"metric_{i}" for i in range(4)]
    df_cls = pd.DataFrame(X_cls, columns=cols_cls)
    cls_path = "static/test_datasets/test_clustering.csv"
    df_cls.to_csv(cls_path, index=False)
    print(f"Generated clustering test data at {cls_path}")

    # 3. Anomaly Dataset
    X_anom, _ = make_blobs(n_samples=190, n_features=3, centers=1, cluster_std=1.0, random_state=42)
    # Add 10 extreme outliers
    np.random.seed(42)
    outliers = np.random.uniform(low=-15, high=15, size=(10, 3))
    X_all = np.vstack([X_anom, outliers])
    cols_anom = ["sensor_0", "sensor_1", "sensor_anomaly_reading"]
    df_anom = pd.DataFrame(X_all, columns=cols_anom)
    anom_path = "static/test_datasets/test_anomaly.csv"
    df_anom.to_csv(anom_path, index=False)
    print(f"Generated anomaly test data at {anom_path}")
    
    return clf_path, cls_path, anom_path

def run_test_session(session_id: str, file_path: str):
    """
    Invoke the LangGraph pipeline programmatically using checkpointer to verify HITL.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    print(f"\n--- Running Verification for Session {session_id} ({os.path.basename(file_path)}) ---")
    
    # We will test the paused and resumed workflow with SqliteSaver checkpointer
    db_path = "static/test_datasets/checkpoints_test.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    with SqliteSaver.from_conn_string(db_path) as memory:
        graph = create_agent_graph(checkpointer=memory)
        config = {"configurable": {"thread_id": session_id}}
        
        initial_state = {
            "session_id": session_id,
            "file_path": file_path,
            "df_shape": [],
            "df_columns": [],
            "df_profile": {},
            "correlation_summary": [],
            "task_type": None,
            "target_column": None,
            "ml_results": None,
            "charts": {},
            "status": "dataset_profiler",
            "reasoning": None,
            "report_path": None,
            "error": None,
            "logs": ["Test session initialized."]
        }
        
        try:
            # 1. Run until the interrupt after ml_task_detector
            print("Running Phase 1 (up to task detection interrupt)...")
            graph.invoke(initial_state, config)
            
            # Verify it paused
            snapshot = graph.get_state(config)
            if not snapshot or not snapshot.next:
                print("Verification FAILED: Pipeline did not pause/interrupt at ml_task_detector.")
                return False
                
            print(f"Phase 1 complete. Status: {snapshot.values.get('status')}")
            print(f"Detected Task Type: {snapshot.values.get('task_type')}")
            print(f"Detected Target Column: {snapshot.values.get('target_column')}")
            
            # 2. Simulate User Approval: override task type and target column if needed, and resume
            current_state = dict(snapshot.values)
            task_type = current_state["task_type"]
            target_column = current_state["target_column"]
            
            current_state["status"] = "model_execution"
            current_state["logs"].append(f"Test simulated user approval of task={task_type}, target={target_column}")
            
            graph.update_state(config, current_state)
            
            print("Running Phase 2 (resuming pipeline from checkpoint)...")
            final_state = graph.invoke(None, config)
            
            # Verify output state parameters
            print(f"Final Status: {final_state.get('status')}")
            if final_state.get('error'):
                print(f"Error encountered: {final_state.get('error')}")
                return False
                
            print(f"Final Task Type: {final_state.get('task_type')}")
            print(f"Final Target Variable: {final_state.get('target_column')}")
            print(f"Generated Charts: {list(final_state.get('charts').keys())}")
            print(f"Generated PDF Report Path: {final_state.get('report_path')}")
            
            # Ensure files actually exist on disk
            report_disk_path = final_state.get('report_path').lstrip('/')
            if not os.path.exists(report_disk_path):
                print(f"Verification FAILED: PDF Report does not exist on disk: {report_disk_path}")
                return False
                
            for chart_rel_path in final_state.get('charts').values():
                chart_disk_path = chart_rel_path.lstrip('/')
                if not os.path.exists(chart_disk_path):
                    print(f"Verification FAILED: Chart does not exist on disk: {chart_disk_path}")
                    return False
                    
            print("Verification SUCCESSful!")
            return True
        except Exception as e:
            print(f"Verification FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    clf_path, cls_path, anom_path = generate_test_data()
    
    # Run tests on the generated files
    success = True
    success &= run_test_session("verify_classification", clf_path)
    success &= run_test_session("verify_clustering", cls_path)
    success &= run_test_session("verify_anomaly", anom_path)
    
    print("\n=================================")
    if success:
        print("ALL PROGRAMMATIC VERIFICATIONS PASSED!")
        sys.exit(0)
    else:
        print("SOME VERIFICATIONS FAILED. PLEASE REVIEW LOGS.")
        sys.exit(1)
