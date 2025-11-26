from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

from .train import trained_models

router = APIRouter()


class UpdateWeightsRequest(BaseModel):
    model_id: str
    weights: Dict[str, float]


class OptimizeWeightsRequest(BaseModel):
    model_id: str
    n_trials: int = 50
    n_jobs: int = -1
    holdout_ratio: float = 0.2
    top_k: int = 10


@router.post("/update-weights")
async def update_weights(request: UpdateWeightsRequest):
    """
    Manually update recommendation weights.
    """
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recommender = trained_models[request.model_id]
    
    try:
        required_keys = {'readiness', 'bayesian', 'utility'}
        if set(request.weights.keys()) != required_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Weights must include: {required_keys}"
            )
        
        total = sum(request.weights.values())
        if total == 0:
            raise HTTPException(status_code=400, detail="Weight sum cannot be zero")
        
        normalized_weights = {k: v / total for k, v in request.weights.items()}
        
        if hasattr(recommender, 'set_weights'):
            recommender.set_weights(normalized_weights)
        else:
            recommender.weights = normalized_weights
        
        return {
            "success": True,
            "weights": normalized_weights,
            "message": "重みを更新しました"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-weights")
async def optimize_weights(request: OptimizeWeightsRequest):
    """
    Automatically optimize weights using Bayesian optimization.
    """
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recommender = trained_models[request.model_id]
    
    if not hasattr(recommender, 'optimize_weights'):
        raise HTTPException(
            status_code=400,
            detail="Model does not support weight optimization"
        )
    
    try:
        best_weights = recommender.optimize_weights(
            n_trials=request.n_trials,
            n_jobs=request.n_jobs,
            holdout_ratio=request.holdout_ratio,
            top_k=request.top_k
        )
        
        return {
            "success": True,
            "optimized_weights": best_weights,
            "message": "重みの最適化が完了しました"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/weights/{model_id}")
async def get_weights(model_id: str):
    """
    Get current weights.
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recommender = trained_models[model_id]
    weights = recommender.get_weights() if hasattr(recommender, 'get_weights') else recommender.weights
    
    return {
        "model_id": model_id,
        "weights": weights
    }
