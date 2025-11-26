from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path

from skillnote_recommendation.graph.causal_graph_visualizer import CausalGraphVisualizer
from .train import trained_models

router = APIRouter()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class EgoGraphRequest(BaseModel):
    model_id: str
    center_node: str
    radius: int = 1
    threshold: float = 0.05
    show_negative: bool = False
    member_skills: Optional[list[str]] = None


class FullGraphRequest(BaseModel):
    model_id: str
    threshold: float = 0.3
    top_n: int = 20
    show_negative: bool = False


@router.post("/graph/ego")
async def get_ego_network(request: EgoGraphRequest):
    """
    Generate ego network graph.
    """
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recommender = trained_models[request.model_id]
    
    try:
        adj_matrix = recommender.learner.get_adjacency_matrix()
        visualizer = CausalGraphVisualizer(adj_matrix)

        output_dir = PROJECT_ROOT / "backend" / "temp_graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ego_{request.model_id}_{request.center_node}.html"
        
        html_path = visualizer.visualize_ego_network_pyvis(
            center_node=request.center_node,
            radius=request.radius,
            threshold=request.threshold,
            show_negative=request.show_negative,
            member_skills=request.member_skills or [],
            output_path=str(output_path),
            height="600px"
        )
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return {
            "success": True,
            "html": html_content
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")


@router.post("/graph/full")
async def get_full_graph(request: FullGraphRequest):
    """
    Generate full causal graph.
    """
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recommender = trained_models[request.model_id]
    
    try:
        adj_matrix = recommender.learner.get_adjacency_matrix()
        visualizer = CausalGraphVisualizer(adj_matrix)

        output_dir = PROJECT_ROOT / "backend" / "temp_graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"full_{request.model_id}.html"
        
        html_path = visualizer.visualize_interactive(
            output_path=str(output_path),
            threshold=request.threshold,
            top_n=request.top_n,
            show_negative=request.show_negative,
            height="800px",
            width="100%"
        )
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return {
            "success": True,
            "html": html_content,
            "node_count": request.top_n
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")
