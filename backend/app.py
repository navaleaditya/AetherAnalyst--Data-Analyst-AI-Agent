import os
import uuid
import shutil
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from celery import Celery

# Load local environment variables (like GEMINI_API_KEY)
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ensure static directories exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)
os.makedirs("static/memory", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

# Shared SQLite DB for checkpointer
CHECKPOINT_DB_PATH = "static/memory/checkpoints.db"

# --- Celery Configuration ---
# Since Redis server is not natively running, we use SQLite as the broker & backend
celery_app = Celery(
    "data_analyst_agent",
    broker="sqla+sqlite:///static/memory/celery_broker.sqlite",
    backend="db+sqlite:///static/memory/celery_results.sqlite"
)

celery_app.conf.update(
    task_always_eager=False,
    broker_connection_retry_on_startup=True
)

app = FastAPI(
    title="Autonomous Data Analyst AI Agent API",
    version="2.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import agent graph creator
from backend.agent import create_agent_graph
from langgraph.checkpoint.sqlite import SqliteSaver

def get_state_from_checkpointer(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        memory = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
        graph = create_agent_graph(checkpointer=memory)
        config = {"configurable": {"thread_id": session_id}}
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return dict(snapshot.values)
    except Exception as e:
        logger.error(f"Failed to read state from checkpointer: {e}")
    return None

# --- Celery Worker Tasks ---

@celery_app.task
def run_agentic_pipeline_task(session_id: str, file_path: str):
    logger.info(f"Starting Celery pipeline task for session {session_id}")
    
    memory = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
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
        "logs": ["Session initialized in Celery worker."]
    }
    
    try:
        # Run graph. It will execute profiler -> correlation -> task detector, then interrupt.
        graph.invoke(initial_state, config)
        
        # Check if paused for human-in-the-loop approval
        snapshot = graph.get_state(config)
        if snapshot and snapshot.next:
            logger.info(f"Pipeline paused after task detector node for session {session_id} - awaiting approval")
            current_state = dict(snapshot.values)
            current_state["status"] = "awaiting_approval"
            current_state["logs"].append("Pipeline paused. Awaiting ML Task & Target column approval.")
            graph.update_state(config, current_state)
        else:
            logger.info(f"Pipeline finished successfully for session {session_id} (no interrupt hit)")
    except Exception as e:
        logger.exception(f"Error in Celery background pipeline for session {session_id}")
        try:
            snapshot = graph.get_state(config)
            current_state = dict(snapshot.values) if (snapshot and snapshot.values) else {}
            current_state["status"] = "error"
            current_state["error"] = str(e)
            if "logs" not in current_state:
                current_state["logs"] = []
            current_state["logs"].append(f"Fatal execution error: {e}")
            graph.update_state(config, current_state)
        except Exception as ex:
            logger.error(f"Failed to record error in SQLite checkpointer: {ex}")

@celery_app.task
def resume_agentic_pipeline_task(session_id: str, task_type: str, target_column: Optional[str]):
    logger.info(f"Resuming Celery pipeline task for session {session_id}")
    
    memory = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
    graph = create_agent_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        snapshot = graph.get_state(config)
        if not snapshot or not snapshot.next:
            logger.warning(f"No paused checkpoint found for session {session_id}")
            return
            
        current_state = dict(snapshot.values)
        current_state["task_type"] = task_type
        current_state["target_column"] = target_column if task_type == "classification" else None
        current_state["status"] = "model_execution"
        current_state["logs"].append(f"User approved ML Task: '{task_type}' and Target Feature: '{target_column}'. Resuming...")
        
        # Save approved configuration in checkpointer state
        graph.update_state(config, current_state)
        
        # Resume loop by invoking with None
        graph.invoke(None, config)
        logger.info(f"Pipeline resumed and completed successfully for session {session_id}")
    except Exception as e:
        logger.exception(f"Error resuming Celery background pipeline for session {session_id}")
        try:
            snapshot = graph.get_state(config)
            current_state = dict(snapshot.values) if (snapshot and snapshot.values) else {}
            current_state["status"] = "error"
            current_state["error"] = str(e)
            if "logs" not in current_state:
                current_state["logs"] = []
            current_state["logs"].append(f"Fatal task resume error: {e}")
            graph.update_state(config, current_state)
        except Exception as ex:
            logger.error(f"Failed to record error on resume: {ex}")

# --- API Endpoints ---

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Ingest a CSV/Excel file, initialize checkpointer state, and dispatch Celery pipeline task.
    """
    filename = file.filename
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported.")
        
    session_id = str(uuid.uuid4())
    file_ext = os.path.splitext(filename)[1]
    saved_filename = f"{session_id}{file_ext}"
    saved_path = os.path.join("static", "uploads", saved_filename)
    
    # Save file on disk
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
        
    # Dispatch execution to Celery background task
    run_agentic_pipeline_task.delay(session_id, saved_path)
    
    return {
        "session_id": session_id,
        "status": "dataset_profiler",
        "message": "Analysis pipeline successfully spawned in background."
    }

class ApprovalPayload(BaseModel):
    task_type: str
    target_column: Optional[str] = None

@app.post("/api/approve/{session_id}")
async def approve_session(session_id: str, payload: ApprovalPayload):
    """
    Receive user ML task and target approval/overrides, then resume Celery worker task.
    """
    state = get_state_from_checkpointer(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    if state.get("status") != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Session is in state '{state.get('status')}', not awaiting_approval.")
        
    # Dispatch resume to Celery worker task
    resume_agentic_pipeline_task.delay(session_id, payload.task_type, payload.target_column)
    
    return {
        "session_id": session_id,
        "status": "model_execution",
        "message": "Approved configurations submitted. Resuming agent execution pipeline."
    }

@app.get("/api/status/{session_id}")
async def get_session_status(session_id: str):
    """
    Poll the current execution status, steps, and logs of a session from the SQLite checkpointer.
    """
    state = get_state_from_checkpointer(session_id)
    if not state:
        # Check if saved in session memory files on disk
        memory_file = os.path.join("static", "memory", f"{session_id}.json")
        if os.path.exists(memory_file):
            import json
            with open(memory_file, 'r', encoding='utf-8') as f:
                cached_state = json.load(f)
                return {
                    "status": cached_state.get("status", "completed"),
                    "error": cached_state.get("error"),
                    "logs": ["Session loaded from cache."],
                    "task_type": cached_state.get("task_type"),
                    "target_column": cached_state.get("target_column"),
                    "df_columns": cached_state.get("df_columns", [])
                }
        raise HTTPException(status_code=404, detail="Session not found.")
        
    return {
        "status": state.get("status"),
        "error": state.get("error"),
        "logs": state.get("logs", []),
        "task_type": state.get("task_type"),
        "target_column": state.get("target_column"),
        "df_columns": state.get("df_columns", [])
    }

@app.get("/api/results/{session_id}")
async def get_session_results(session_id: str):
    """
    Retrieve full execution results (profile, correlations, model stats, and chart image mappings).
    """
    state = get_state_from_checkpointer(session_id)
    if not state:
        memory_file = os.path.join("static", "memory", f"{session_id}.json")
        if os.path.exists(memory_file):
            import json
            with open(memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="Session not found.")
        
    return state

@app.get("/api/report/{session_id}")
async def get_session_report(session_id: str):
    """
    Download the generated PDF report.
    """
    report_name = f"report_{session_id}.pdf"
    report_path = os.path.join("static", "reports", report_name)
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not generated yet or session expired.")
        
    return FileResponse(
        report_path, 
        media_type="application/pdf", 
        filename=f"Data_Analysis_Report_{session_id[:8]}.pdf"
    )

# --- Frontend Static Routes ---

@app.get("/")
async def get_frontend_index():
    """
    Serve the main SPA page.
    """
    index_path = "frontend/index.html"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)

# Serve static files inside frontend directory (CSS/JS files)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    logger.info("Launching FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
