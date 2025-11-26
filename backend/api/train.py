from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
import time
import os
import pandas as pd
from pathlib import Path

from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender

router = APIRouter()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Simple in-memory storage for trained models
trained_models: Dict[str, CausalGraphRecommender] = {}


class TrainRequest(BaseModel):
    session_id: str
    min_members_per_skill: int = 5
    correlation_threshold: float = 0.2
    weights: Optional[Dict[str, float]] = None


class TrainResponse(BaseModel):
    success: bool
    model_id: str
    summary: Dict[str, Any]
    message: str


@router.post("/train", response_model=TrainResponse)
async def train_causal_model(request: TrainRequest):
    """
    Train a causal model using LiNGAM.
    """
    try:
        # Get upload directory for this session
        base_upload_dir = PROJECT_ROOT / "backend" / "temp_uploads"
        session_dir = base_upload_dir / request.session_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session data not found. Please upload data first.")
        
        print(f"[TRAIN] Loading data from {session_dir}")
        start_load = time.time()
        
        # Load CSV files directly
        data = {}
        csv_files = {
            'members': 'members.csv',
            'skills': 'skills.csv',
            'education': 'education.csv',
            'license': 'license.csv',
            'categories': 'categories.csv',
            'acquired': 'acquired.csv'
        }
        
        for key, filename in csv_files.items():
            filepath = session_dir / filename
            if not filepath.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {filename}")
            data[key] = pd.read_csv(filepath)
        
        print(f"[TRAIN] Data loaded in {time.time() - start_load:.2f}s")
        
        # Transform data
        start_transform = time.time()
        transformer = DataTransformer()
        
        competence_master = transformer.create_competence_master(data)
        member_competence, valid_members = transformer.create_member_competence(data, competence_master)
        
        print(f"[TRAIN] Data transformed in {time.time() - start_transform:.2f}s")
        
        # Prepare transformed data
        transformed_data = {
            "member_competence": member_competence,
            "competence_master": competence_master
        }
        
        # Default weights
        weights = request.weights or {
            'readiness': 0.6,
            'bayesian': 0.3,
            'utility': 0.1
        }
        
        # Start training
        start_train = time.time()
        print(f"[TRAIN] Starting model training...")
        
        recommender = CausalGraphRecommender(
            member_competence=transformed_data["member_competence"],
            competence_master=transformed_data["competence_master"],
            learner_params={
                "correlation_threshold": request.correlation_threshold,
                "min_cluster_size": 3
            },
            weights=weights
        )
        
        recommender.fit(min_members_per_skill=request.min_members_per_skill)
        
        learning_time = time.time() - start_train
        print(f"[TRAIN] Model training completed in {learning_time:.2f}s")
        
        # Generate model ID
        model_id = f"model_{request.session_id}_{int(time.time())}"
        
        # Store trained model
        trained_models[model_id] = recommender
        
        # Prepare summary
        summary = {
            "num_members": len(recommender.skill_matrix_.index),
            "num_skills": len(recommender.skill_matrix_.columns),
            "learning_time": round(learning_time, 2),
            "weights": weights
        }
        
        return TrainResponse(
            success=True,
            model_id=model_id,
            summary=summary,
            message="因果構造の学習が完了しました"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/model/{model_id}/summary")
async def get_model_summary(model_id: str):
    """
    Get summary of a trained model.
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recommender = trained_models[model_id]
    weights = recommender.get_weights() if hasattr(recommender, 'get_weights') else recommender.weights
    
    return {
        "model_id": model_id,
        "num_members": len(recommender.skill_matrix_.index),
        "num_skills": len(recommender.skill_matrix_.columns),
        "weights": weights
    }
