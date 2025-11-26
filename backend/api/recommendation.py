from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from .train import trained_models

router = APIRouter()


class RecommendationRequest(BaseModel):
    model_id: str
    member_id: str
    top_n: int = 10


@router.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """
    Get skill recommendations for a member using a trained model.
    """
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found. Please train a model first.")
    
    recommender = trained_models[request.model_id]
    
    try:
        recommendations = recommender.recommend(request.member_id, top_n=request.top_n)
        
        if not recommendations:
            return {
                "member_id": request.member_id,
                "recommendations": [],
                "message": "推奨できるスキルが見つかりませんでした"
            }
        
        return {
            "member_id": request.member_id,
            "recommendations": recommendations
        }
        
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Member ID '{request.member_id}' not found in the training data"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
