"""
Constraint management API endpoints.

This module provides endpoints for managing causal graph constraints,
allowing users to enforce domain knowledge on learned causal structures.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.request.constraint import AddConstraintRequest, ApplyConstraintsRequest
from backend.schemas.response.constraint import (
    GetConstraintsResponse,
    AddConstraintResponse,
    DeleteConstraintResponse,
    ApplyConstraintsResponse,
    Constraint,
)
from backend.utils.constraint_manager import constraint_manager
from backend.utils import session_manager
from backend.core.exceptions import ModelNotFoundException
from backend.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/constraints/{session_id}", response_model=GetConstraintsResponse)
async def get_constraints(session_id: str):
    """
    Get all constraints for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of constraints
    """
    try:
        constraints = constraint_manager.load_constraints(session_id)
        
        return GetConstraintsResponse(
            session_id=session_id,
            constraints=[Constraint(**c) for c in constraints],
            count=len(constraints)
        )
        
    except Exception as e:
        logger.error("Failed to get constraints", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"制約の読み込みに失敗しました: {str(e)}")


@router.post("/constraints/{session_id}", response_model=AddConstraintResponse)
async def add_constraint(session_id: str, request: AddConstraintRequest):
    """
    Add a new constraint to a session's causal graph.
    
    Args:
        session_id: Session identifier
        request: Constraint details
        
    Returns:
        The created constraint
    """
    try:
        # Validate constraint type
        if request.constraint_type not in ['required', 'forbidden', 'deleted']:
            raise HTTPException(
                status_code=400,
                detail=f"無効な制約タイプ: {request.constraint_type}. " 
                       "'required', 'forbidden', 'deleted'のいずれかを指定してください"
            )
        
        constraint = constraint_manager.add_constraint(
            session_id=session_id,
            from_skill=request.from_skill,
            to_skill=request.to_skill,
            constraint_type=request.constraint_type,
            value=request.value,
            user_id=request.user_id
        )
        
        return AddConstraintResponse(
            constraint=Constraint(**constraint)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add constraint", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"制約の追加に失敗しました: {str(e)}")


@router.delete("/constraints/{session_id}/{constraint_id}", response_model=DeleteConstraintResponse)
async def delete_constraint(session_id: str, constraint_id: str):
    """
    Delete a constraint by ID.
    
    Args:
        session_id: Session identifier
        constraint_id: Constraint identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        success = constraint_manager.delete_constraint(session_id, constraint_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"制約が見つかりません: {constraint_id}"
            )
        
        return DeleteConstraintResponse()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete constraint", error=str(e))
        raise HTTPException(status_code=500, detail=f"制約の削除に失敗しました: {str(e)}")


@router.post("/constraints/{session_id}/apply", response_model=ApplyConstraintsResponse)
async def apply_constraints(session_id: str, request: ApplyConstraintsRequest):
    """
    Apply constraints to an existing model without retraining.
    
    Args:
        session_id: Session identifier
        request: Model ID to apply constraints to
        
    Returns:
        Application result
    """
    try:
        # Get model
        recommender = session_manager.get_model(request.model_id)
        
        if not recommender:
            raise ModelNotFoundException(request.model_id)
        
        # Load constraints
        constraints = constraint_manager.load_constraints(session_id)
        
        if not constraints:
            logger.info("No constraints to apply", session_id=session_id)
            return ApplyConstraintsResponse(
                model_id=request.model_id,
                applied_count=0,
                skipped_count=0,
                message="制約がありません"
            )
        
        # Apply constraints
        recommender.learner.apply_constraints(constraints)
        
        # Count results (simple approximation)
        applied_count = len(constraints)
        skipped_count = 0
        
        logger.info(
            "Applied constraints to model",
            model_id=request.model_id,
            applied=applied_count
        )
        
        return ApplyConstraintsResponse(
            model_id=request.model_id,
            applied_count=applied_count,
            skipped_count=skipped_count
        )
        
    except ModelNotFoundException:
        raise HTTPException(
            status_code=404,
            detail=f"モデルが見つかりません: {request.model_id}"
        )
    except Exception as e:
        logger.error("Failed to apply constraints", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"制約の適用に失敗しました: {str(e)}")
