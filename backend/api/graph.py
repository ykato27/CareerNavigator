from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from backend.utils import PROJECT_ROOT, session_manager
from skillnote_recommendation.graph.causal_graph_visualizer import CausalGraphVisualizer

router = APIRouter()
logger = logging.getLogger(__name__)


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
    Generate ego network graph visualization.

    Args:
        request: Graph parameters including model_id, center_node, radius, etc.

    Returns:
        dict: HTML content of the interactive graph

    Raises:
        HTTPException: If model not found or graph generation fails
    """
    # Get model from session manager
    recommender = session_manager.get_model(request.model_id)

    if not recommender:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Please train a model first.",
        )

    try:
        logger.info(f"[GRAPH] Generating ego network for {request.center_node}")

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
            height="600px",
        )

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        logger.info("[GRAPH] Ego network generated successfully")

        return {"success": True, "html": html_content}

    except Exception as e:
        logger.error(f"[GRAPH] Graph generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")


@router.post("/graph/full")
async def get_full_graph(request: FullGraphRequest):
    """
    Generate full causal graph visualization.

    Args:
        request: Graph parameters including model_id, threshold, top_n, etc.

    Returns:
        dict: HTML content of the interactive graph

    Raises:
        HTTPException: If model not found or graph generation fails
    """
    # Get model from session manager
    recommender = session_manager.get_model(request.model_id)

    if not recommender:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Please train a model first.",
        )

    try:
        logger.info(f"[GRAPH] Generating full causal graph (top {request.top_n})")

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
            width="100%",
        )

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        logger.info("[GRAPH] Full causal graph generated successfully")

        return {"success": True, "html": html_content, "node_count": request.top_n}

    except Exception as e:
        logger.error(f"[GRAPH] Graph generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")
