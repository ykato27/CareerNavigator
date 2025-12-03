from fastapi import APIRouter, UploadFile, HTTPException, Request
from typing import List
import time
import pandas as pd
import io

from backend.utils import get_upload_dir, session_manager, load_csv_files, clean_dataframe_columns

router = APIRouter()


def merge_csv_files(files: List[UploadFile], category_name: str) -> tuple[pd.DataFrame, List[str]]:
    """
    Merge multiple CSV files and check for duplicate rows.

    Args:
        files: List of uploaded CSV files
        category_name: Name of the category (for error messages)

    Returns:
        tuple: (merged DataFrame, list of duplicate row descriptions)

    Raises:
        ValueError: If duplicate rows found
    """
    dfs = []
    duplicate_errors = []

    for idx, file in enumerate(files):
        df = pd.read_csv(io.BytesIO(file), encoding="utf-8-sig")
        df["_source_file"] = file.filename or f"file_{idx}"
        df["_source_row"] = range(2, len(df) + 2)  # Row numbers starting from 2 (after header)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No files provided for {category_name}")

    # Concatenate all dataframes
    merged = pd.concat(dfs, ignore_index=True)

    # Check for duplicate rows (excluding metadata columns)
    data_columns = [col for col in merged.columns if not col.startswith("_")]
    duplicates = merged[merged.duplicated(subset=data_columns, keep=False)]

    if not duplicates.empty:
        # Group duplicates and create error messages
        for _, group in duplicates.groupby(data_columns):
            if len(group) > 1:
                file_info = []
                for _, row in group.iterrows():
                    file_info.append(f"{row['_source_file']}の{row['_source_row']}行目")
                duplicate_errors.append(f"重複データ: {', '.join(file_info)}")

    # Remove metadata columns before returning
    merged = merged[data_columns]

    return merged, duplicate_errors


@router.post("/upload")
async def upload_files(request: Request):
    """
    Upload multiple CSV files per category for career analysis.
    Supports multiple files per category which will be merged.

    Returns:
        dict: Contains session_id, message, and list of uploaded files
    """
    try:
        # Parse multipart form data
        form = await request.form()

        # Organize files by category
        files_by_category = {
            "members": [],
            "skills": [],
            "education": [],
            "license": [],
            "categories": [],
            "acquired": [],
        }

        # Group files by category
        for key, value in form.items():
            # Extract category name (e.g., "members[0]" -> "members")
            if "[" in key:
                category = key.split("[")[0]
                if category in files_by_category and hasattr(value, "read"):
                    files_by_category[category].append(value)

        # Validate all categories have at least one file
        missing_categories = [cat for cat, files in files_by_category.items() if len(files) == 0]
        if missing_categories:
            raise HTTPException(
                status_code=400,
                detail=f"以下のカテゴリにファイルがありません: {', '.join(missing_categories)}",
            )

        # Create session ID with timestamp
        session_id = f"session_{int(time.time())}"

        # Get upload directory for this session
        upload_dir = get_upload_dir(session_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        all_duplicate_errors = []

        # Process and merge files for each category
        for category, file_list in files_by_category.items():
            if len(file_list) == 1:
                # Single file - just save it
                file_path = upload_dir / f"{category}.csv"
                content = await file_list[0].read()
                with open(file_path, "wb") as f:
                    f.write(content)
            else:
                # Multiple files - merge them
                file_contents = []
                for file in file_list:
                    content = await file.read()
                    file_contents.append(content)
                    await file.seek(0)  # Reset for potential re-read

                # Create temporary file objects with content
                temp_files = []
                for content, file in zip(file_contents, file_list):
                    temp_file = io.BytesIO(content)
                    temp_file.filename = file.filename
                    temp_files.append(temp_file)

                # Merge and check for duplicates
                merged_df, duplicate_errors = merge_csv_files(temp_files, category)

                if duplicate_errors:
                    all_duplicate_errors.extend([f"[{category}] {err}" for err in duplicate_errors])

                # Save merged file
                file_path = upload_dir / f"{category}.csv"
                merged_df.to_csv(file_path, index=False, encoding="utf-8-sig")

        # If there are duplicate errors, raise exception
        if all_duplicate_errors:
            raise HTTPException(
                status_code=400,
                detail="重複行が検出されました:\n" + "\n".join(all_duplicate_errors),
            )

        # Register session in session manager
        session_manager.add_session(
            session_id, {"dir": str(upload_dir), "files": list(files_by_category.keys())}
        )

        return {
            "session_id": session_id,
            "message": "Files uploaded and merged successfully",
            "files": list(files_by_category.keys()),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


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
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "exists": True,
        "files": session_data.get("files", []),
        "timestamp": session_data.get("timestamp"),
        "last_accessed": session_data.get("last_accessed"),
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
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove files from disk
    upload_dir = get_upload_dir(session_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    return {"success": True, "message": f"Session {session_id} deleted successfully"}


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
        "max_age_hours": max_age_hours,
    }


@router.get("/session/{session_id}/members")
async def get_session_members(session_id: str):
    """
    Get list of members from uploaded data.

    Args:
        session_id: The session identifier

    Returns:
        dict: List of members with their codes and names

    Raises:
        HTTPException: If session not found
    """
    try:
        # Load CSV files
        data = load_csv_files(session_id)

        # Get members data and clean column names
        members_df = clean_dataframe_columns(data["members"])

        # Extract member information
        members_list = []
        for idx, row in members_df.iterrows():
            member_code = str(row.get("メンバーコード", ""))
            member_name = str(row.get("メンバー名", ""))

            if member_code and member_name:
                members_list.append(
                    {
                        "member_code": member_code,
                        "member_name": member_name,
                        "display_name": f"{member_code} - {member_name}",
                    }
                )

        # Sort by member code
        members_list.sort(key=lambda x: x["member_code"])

        return {"success": True, "members": members_list, "total_count": len(members_list)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get members list: {str(e)}")
