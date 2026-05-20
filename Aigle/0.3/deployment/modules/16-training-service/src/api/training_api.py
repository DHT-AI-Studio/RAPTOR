"""Training API endpoints for job submission and management.

This module provides REST API endpoints for:
- Submitting training jobs
- Tracking job status and progress
- Listing and filtering jobs
- Cancelling running jobs
- Deleting job records
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from pydantic import BaseModel, Field

from ..core.training_job_manager import get_job_manager, TrainingJobManager
from ..training.models.job_submission import TrainingJobSubmission


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/training",
    tags=["Training Job Management"]
)


def get_manager() -> TrainingJobManager:
    """Get the global TrainingJobManager instance."""
    return get_job_manager()


# --- Pydantic Models for Responses ---

class JobStatusResponse(BaseModel):
    """Single training job status response model."""
    
    job_id: str = Field(..., description="Unique job identifier")
    gpu_id: int | List[int] = Field(None, description="Assigned GPU ID(s)")
    status: str = Field(..., description="Job status (queued, running, completed, failed, cancelled)")
    config: Optional[dict] = Field(None, description="Training configuration details")
    start_time: Optional[str] = Field(None, description="Start time (ISO 8601)")
    end_time: Optional[str] = Field(None, description="End time (ISO 8601)")
    metrics: Optional[dict] = Field(None, description="Final training metrics or error information")
    model_path: Optional[str] = Field(None, description="Path to trained model after completion")


# --- API Endpoints ---

@router.post("/submit", response_model=JobStatusResponse, summary="Submit a new training job")
def submit_training_job_endpoint(
    submission: TrainingJobSubmission,
    job_manager: TrainingJobManager = Depends(get_manager)
):
    """
    Submit a new AI model training job to the scheduler.
    
    For Instruction Tuning / SFT tasks, the system supports multiple column combinations
    for providing Instruction Dataset. All columns are automatically converted to 
    standardized Chat Messages format via `column_mapping`.
    
    **Supported columns (via column_mapping):**
    - `messages`: Complete multi-turn conversation (used directly if present)
    - `reasoning`: Model reasoning process (CoT)
    - `context`: Context/background information
    - `input`: Instruction, question, or prompt text
    - `output`: Expected model response
    
    **Single-turn data auto-conversion format:**
    If the data does not provide a `messages` column, the system automatically 
    assembles the following conversation:
    - system: The `default_system_prompt` from config or data's `system_prompt`
    - user: 
        - With `context` → `Context: {context} Question/Instruction: {input}`
        - Otherwise → `{input}`
    - assistant:
        - With `reasoning` → Wrapped as:
        
             ```
             {reasoning}
             ```
             
             {output}
        
        - Otherwise → Directly `{output}`
    
    **Multi-turn messages format requirements:**
    - Each item must contain: `{"role": "system|user|assistant", "content": str}`
    - Empty content or invalid roles are automatically cleaned
    - The last assistant response is automatically appended with EOS token (if tokenizer has it defined)
    
    For Text Generation tasks, the `text` column can be specified via `column_mapping` 
    to provide pure text generation data.
    """
    try:
        job_id = job_manager.submit_job(submission)
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=500, detail="Job submission failed internally")
            
        return job.to_dict()  # Return full job details
        
    except Exception as e:
        logger.error(f"Failed to submit job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.get("/status/{job_id}", response_model=JobStatusResponse, summary="Get detailed status of a specific job")
async def get_job_status_endpoint(
    job_id: str = Path(..., description="Job ID to query"),
    job_manager: TrainingJobManager = Depends(get_manager)
):
    """
    Query the current status, assigned GPU, and results of a specified training job.
    
    This endpoint returns:
    - Basic job information (ID, status, GPU assignment)
    - Real-time progress metrics from MLflow (for running jobs)
    - Final metrics and model path (for completed jobs)
    """
    # 1. Get real-time progress (queries MLflow)
    progress_data = job_manager.get_job_progress(job_id)

    # 2. Get basic/final job status (from Redis)
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found")
    
    response_data = job.to_dict()
    
    # Add progress data for non-completed jobs
    if job.status != "completed" and progress_data:
        if response_data["metrics"] is None:
            response_data["metrics"] = {}
        response_data["metrics"].update(progress_data)
        
    return response_data


@router.get("/list", response_model=List[JobStatusResponse], summary="List all training jobs")
def list_all_jobs_endpoint(
    status: Optional[str] = Query(None, description="Filter by status (queued, running, completed, failed, cancelled)"),
    job_manager: TrainingJobManager = Depends(get_manager)
):
    """
    List all submitted training jobs, optionally filtered by status.
    
    Returns full job details including metrics and progress information.
    """
    try:
        jobs_list = job_manager.list_jobs(status=status)
        
        # Convert summary list to full JobStatusResponse list
        full_jobs_list = []
        for job_summary in jobs_list:
            # Load full Job object
            job = job_manager.get_job(job_summary['job_id'])
            progress_data = job_manager.get_job_progress(job_summary['job_id'])

            response_data = job.to_dict()
            
            # Add progress data for non-completed jobs
            if job.status != "completed" and progress_data:
                if response_data["metrics"] is None:
                    response_data["metrics"] = {}
                response_data["metrics"].update(progress_data) 

            full_jobs_list.append(response_data)
            
        return full_jobs_list
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.post("/cancel/{job_id}", summary="Cancel a running or queued training job")
def cancel_job_endpoint(
    job_id: str = Path(..., description="Job ID to cancel"),
    job_manager: TrainingJobManager = Depends(get_manager)
):
    """
    Attempt to cancel a specified training job.
    
    - If the job is in 'queued' status, it will be immediately removed from the queue.
    - If the job is in 'running' status, it will be marked as 'cancelled'. 
      The training process will gracefully shutdown at the next checkpoint.
    """
    success = job_manager.cancel_job(job_id)
    
    if not success:
        # Check if job exists or is already in terminal state
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found")
        
        if job.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Job '{job_id}' is already in status: {job.status}. Cannot cancel."
            )

        raise HTTPException(status_code=500, detail="Failed to cancel job for unknown reason")
        
    return {"message": f"Cancellation request processed for Job ID: {job_id}"}


@router.delete("/delete/{job_id}", summary="Delete job records")
def delete_job_endpoint(
    job_id: str = Path(..., description="Job ID to delete"),
    force: bool = Query(False, description="Force cancel and delete if job is still running or queued"),
    job_manager: TrainingJobManager = Depends(get_manager)
):
    """
    Delete job records from Redis and clean up associated files.
    
    - If the job is still running or queued, you must set `force=true` to delete.
    - This will remove records from Redis and delete logs and checkpoints from disk.
    
    **Warning:** This operation cannot be undone.
    """
    try:
        deleted = job_manager.delete_job(job_id, force_cancel=force)
        
        if deleted:
            return {"message": f"Job ID '{job_id}' successfully deleted"}
        else:
            job = job_manager.get_job(job_id)
            
            if job and job.status in ["running", "queued"] and not force:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Job '{job_id}' is still {job.status}. Use 'force=true' to cancel and delete."
                )
            
            raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found or could not be deleted")
            
    except HTTPException:
        # Re-raise 400 errors
        raise
    except Exception as e:
        logger.error(f"Error during deletion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {str(e)}")
