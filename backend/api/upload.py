from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import os
from pathlib import Path
import time

router = APIRouter()

# Session storage
SESSION_STORAGE: Dict = {}

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


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
    """
    try:
        # Create session ID
        session_id = f"session_{int(time.time())}"
        
        # Create upload directory
        upload_dir = PROJECT_ROOT / "backend" / "temp_uploads" / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files
        files_map = {
            "members": members,
            "skills": skills,
            "education": education,
            "license": license,
            "categories": categories,
            "acquired": acquired
        }
        
        for key, file in files_map.items():
            file_path = upload_dir / f"{key}.csv"
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
        
        # Store session info
        SESSION_STORAGE[session_id] = {
            "dir": str(upload_dir),
            "timestamp": time.time()
        }
        
        return {
            "session_id": session_id,
            "message": "Files uploaded successfully",
            "files": list(files_map.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
