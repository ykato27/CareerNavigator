from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import time

from backend.utils import PROJECT_ROOT, get_upload_dir, session_manager

router = APIRouter()


@router.post("/upload")
async def upload_files(
    members: UploadFile = File(...),
    skills: UploadFile = File(...),
    education: UploadFile = File(...),
    license: UploadFile = File(...),
    categories: UploadFile = File(...),
    acquired: UploadFile = File(...)
):
    """
    Upload 6 CSV files for career analysis.

    Returns:
        dict: Contains session_id, message, and list of uploaded files
    """
    try:
        # Create session ID with timestamp
        session_id = f"session_{int(time.time())}"

        # Get upload directory for this session
        upload_dir = get_upload_dir(session_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Define file mapping
        files_map = {
            "members": members,
            "skills": skills,
            "education": education,
            "license": license,
            "categories": categories,
            "acquired": acquired
        }

        # Save all files
        for key, file in files_map.items():
            file_path = upload_dir / f"{key}.csv"
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

        # Register session in session manager
        session_manager.add_session(session_id, {
            "dir": str(upload_dir),
            "files": list(files_map.keys())
        })

        return {
            "session_id": session_id,
            "message": "Files uploaded successfully",
            "files": list(files_map.keys())
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )


@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get status of a session.

    Args:
        session_id: The session identifier

    Returns:
        dict: Session status information
    """
    session_data = session_manager.get_session(session_id)

    if not session_data:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return {
        "session_id": session_id,
        "exists": True,
        "files": session_data.get("files", []),
        "timestamp": session_data.get("timestamp"),
        "last_accessed": session_data.get("last_accessed")
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its data.

    Args:
        session_id: The session identifier

    Returns:
        dict: Deletion status
    """
    import shutil

    # Remove from session manager
    removed = session_manager.remove_session(session_id)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    # Remove files from disk
    upload_dir = get_upload_dir(session_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    return {
        "success": True,
        "message": f"Session {session_id} deleted successfully"
    }


@router.get("/sessions/stats")
async def get_sessions_stats():
    """
    Get statistics about active sessions and models.

    Returns:
        dict: Statistics including session count, model count, cache size
    """
    stats = session_manager.get_stats()
    return stats


@router.post("/sessions/cleanup")
async def cleanup_old_sessions(max_age_hours: int = 24):
    """
    Clean up old sessions and models.

    Args:
        max_age_hours: Maximum age in hours (default: 24)

    Returns:
        dict: Number of sessions and models removed
    """
    max_age_seconds = max_age_hours * 3600

    sessions_removed = session_manager.cleanup_old_sessions(max_age_seconds)
    models_removed = session_manager.cleanup_old_models(max_age_seconds)

    return {
        "success": True,
        "sessions_removed": sessions_removed,
        "models_removed": models_removed,
        "max_age_hours": max_age_hours
    }
