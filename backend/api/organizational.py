from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging

from backend.utils import load_and_transform_session_data, session_manager
from skillnote_recommendation.organizational.skill_gap_analyzer import SkillGapAnalyzer
from skillnote_recommendation.organizational import org_metrics
from skillnote_recommendation.strategic.succession_planner import SuccessionPlanner
from skillnote_recommendation.strategic.org_simulator import OrganizationSimulator

router = APIRouter()
logger = logging.getLogger(__name__)


class OrganizationalMetricsRequest(BaseModel):
    session_id: str


class SkillGapAnalysisRequest(BaseModel):
    session_id: str
    percentile: float = 0.2


class SuccessionPlanningRequest(BaseModel):
    session_id: str
    target_position: str
    max_candidates: int = 20


class OrganizationSimulationRequest(BaseModel):
    session_id: str
    transfers: List[Dict[str, str]]  # [{member_code, from_group, to_group}]
    group_column: str = "職種"


@router.post("/organizational/metrics")
async def get_organizational_metrics(request: OrganizationalMetricsRequest):
    """
    Get organizational skill map metrics and dashboard data.

    Args:
        request: Contains session_id

    Returns:
        dict: Organizational metrics including coverage, diversity, and top skills

    Raises:
        HTTPException: If session not found or analysis fails
    """
    try:
        logger.info(f"[ORG] Calculating metrics for session {request.session_id}")

        # Load transformed data (with caching via session_manager)
        cache_key = f"org_data_{request.session_id}"
        transformed_data = session_manager.get_cache(cache_key)

        if not transformed_data:
            transformed_data = load_and_transform_session_data(request.session_id)
            session_manager.add_cache(cache_key, transformed_data)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]
        members_df = transformed_data["members_clean"]

        # Calculate KPI metrics
        total_members = len(members_df)
        total_skills = len(competence_master_df)
        total_skill_records = len(member_competence_df)
        avg_skills_per_member = total_skill_records / total_members if total_members > 0 else 0

        # Calculate organizational metrics
        coverage_info = org_metrics.calculate_skill_coverage(
            member_competence_df, competence_master_df
        )
        concentration_info = org_metrics.calculate_skill_concentration(
            member_competence_df, threshold=3
        )
        diversity_index = org_metrics.calculate_skill_diversity_index(member_competence_df)

        # Get skill distribution by category
        category_distribution = {}
        if "力量カテゴリ" in competence_master_df.columns:
            category_counts = competence_master_df["力量カテゴリ"].value_counts().to_dict()
            category_distribution = {str(k): int(v) for k, v in category_counts.items()}

        # Get top skills by member count
        skill_counts = member_competence_df["力量コード"].value_counts().head(10)
        top_skills = []
        for skill_code, count in skill_counts.items():
            skill_info = competence_master_df[competence_master_df["力量コード"] == skill_code]
            if not skill_info.empty:
                top_skills.append(
                    {
                        "skill_code": skill_code,
                        "skill_name": skill_info.iloc[0]["力量名"],
                        "member_count": int(count),
                    }
                )

        logger.info("[ORG] Metrics calculated successfully")

        return {
            "success": True,
            "metrics": {
                "total_members": total_members,
                "total_skills": total_skills,
                "avg_skills_per_member": round(avg_skills_per_member, 1),
                "coverage_rate": round(coverage_info.get("coverage_rate", 0), 3),
                "diversity_index": round(diversity_index, 2),
                "high_concentration_skills": concentration_info.get("high_concentration_skills", 0),
                "low_concentration_skills": concentration_info.get("low_concentration_skills", 0),
            },
            "category_distribution": category_distribution,
            "top_skills": top_skills,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ORG] Metrics calculation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get organizational metrics: {str(e)}"
        )


@router.post("/organizational/skill-gap")
async def analyze_skill_gap(request: SkillGapAnalysisRequest):
    """
    Analyze skill gaps using top percentile approach.

    Args:
        request: Contains session_id and percentile (default: 0.2)

    Returns:
        dict: Skill gap analysis with critical skills identified

    Raises:
        HTTPException: If session not found or analysis fails
    """
    try:
        logger.info(f"[ORG] Analyzing skill gap for session {request.session_id}")

        # Load transformed data (with caching)
        cache_key = f"org_data_{request.session_id}"
        transformed_data = session_manager.get_cache(cache_key)

        if not transformed_data:
            transformed_data = load_and_transform_session_data(request.session_id)
            session_manager.add_cache(cache_key, transformed_data)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]

        # Initialize analyzer
        analyzer = SkillGapAnalyzer()

        # Calculate profiles and gap
        current_profile = analyzer.calculate_current_profile(
            member_competence_df, competence_master_df
        )
        target_profile = analyzer.calculate_target_profile_top_percentile(
            member_competence_df, competence_master_df, percentile=request.percentile
        )
        gap_df = analyzer.calculate_gap(current_profile, target_profile)

        # Identify critical skills
        critical_skills = analyzer.identify_critical_skills(gap_df, threshold=0.3)

        # Format results
        gap_data = _format_gap_dataframe(gap_df.head(20))
        critical_skills_data = _format_gap_dataframe(critical_skills.head(10))

        logger.info("[ORG] Gap analysis completed successfully")

        return {
            "success": True,
            "percentile": request.percentile,
            "gap_analysis": gap_data,
            "critical_skills": critical_skills_data,
            "summary": {
                "total_skills_analyzed": len(gap_df),
                "critical_skills_count": len(critical_skills),
                "avg_gap_rate": round(gap_df["保有率ギャップ率"].mean(), 3),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ORG] Skill gap analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Skill gap analysis failed: {str(e)}")


@router.post("/organizational/succession")
async def find_succession_candidates(request: SuccessionPlanningRequest):
    """
    Find succession candidates for a target position.

    Args:
        request: Contains session_id, target_position, and max_candidates

    Returns:
        dict: Succession candidates ranked by readiness score

    Raises:
        HTTPException: If session not found or search fails
    """
    try:
        logger.info(
            f"[ORG] Finding succession candidates for {request.target_position} "
            f"in session {request.session_id}"
        )

        # Load transformed data (with caching)
        cache_key = f"org_data_{request.session_id}"
        transformed_data = session_manager.get_cache(cache_key)

        if not transformed_data:
            transformed_data = load_and_transform_session_data(request.session_id)
            session_manager.add_cache(cache_key, transformed_data)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]
        members_df = transformed_data["members_clean"]

        # Initialize succession planner
        planner = SuccessionPlanner()

        # Calculate position profile and find candidates
        profile = planner.calculate_position_skill_profile(
            request.target_position,
            members_df,
            member_competence_df,
            competence_master_df,
            position_column="役職",
        )
        candidates = planner.find_succession_candidates(
            request.target_position,
            members_df,
            member_competence_df,
            competence_master_df,
            position_column="役職",
            grade_column="職能・等級",
            exclude_current_holders=True,
            max_candidates=request.max_candidates,
        )

        # Format candidates
        candidates_data = _format_succession_candidates(candidates, planner, request.max_candidates)

        logger.info(f"[ORG] Found {len(candidates_data)} succession candidates")

        return {
            "success": True,
            "target_position": request.target_position,
            "candidates": candidates_data,
            "profile": {"required_skills_count": len(profile) if profile is not None else 0},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ORG] Succession planning failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Succession planning failed: {str(e)}")


@router.post("/organizational/simulate")
async def simulate_organization_changes(request: OrganizationSimulationRequest):
    """
    Simulate organizational changes (member transfers).

    Args:
        request: Contains session_id, transfers list, and group_column

    Returns:
        dict: Simulation results with before/after comparison

    Raises:
        HTTPException: If session not found or simulation fails
    """
    try:
        logger.info(
            f"[ORG] Simulating {len(request.transfers)} transfers for session {request.session_id}"
        )

        # Load transformed data (with caching)
        cache_key = f"org_data_{request.session_id}"
        transformed_data = session_manager.get_cache(cache_key)

        if not transformed_data:
            transformed_data = load_and_transform_session_data(request.session_id)
            session_manager.add_cache(cache_key, transformed_data)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]
        members_df = transformed_data["members_clean"]

        # Initialize simulator
        simulator = OrganizationSimulator()

        # Capture current state
        current_state = simulator.capture_current_state(
            members_df, member_competence_df, competence_master_df, group_by=request.group_column
        )

        # Apply transfers
        for transfer in request.transfers:
            simulator.simulate_transfer(
                transfer["member_code"],
                transfer["from_group"],
                transfer["to_group"],
                group_column=request.group_column,
            )

        # Execute simulation
        simulated_state = simulator.execute_simulation(competence_master_df)

        # Compare states
        comparison_df = simulator.compare_states()

        # Calculate balance scores
        current_balance = simulator.calculate_balance_score(current_state)
        simulated_balance = simulator.calculate_balance_score(simulated_state)

        # Format comparison data
        comparison_data = _format_simulation_comparison(comparison_df)

        logger.info("[ORG] Simulation completed successfully")

        return {
            "success": True,
            "transfers_applied": len(request.transfers),
            "comparison": comparison_data,
            "balance_scores": {
                "current": round(current_balance, 3),
                "simulated": round(simulated_balance, 3),
                "change": round(simulated_balance - current_balance, 3),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ORG] Organization simulation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Organization simulation failed: {str(e)}")


# Helper functions
def _format_gap_dataframe(df) -> List[Dict]:
    """Format gap analysis DataFrame for JSON response."""
    result = []
    for idx, row in df.iterrows():
        result.append(
            {
                "skill_code": row["力量コード"],
                "skill_name": row["力量名"],
                "current_rate": round(row["現在保有率"], 3),
                "target_rate": round(row["目標保有率"], 3),
                "gap_rate": round(row["保有率ギャップ"], 3),
                "gap_percentage": round(row["保有率ギャップ率"], 3),
            }
        )
    return result


def _format_succession_candidates(candidates, planner, max_count: int) -> List[Dict]:
    """Format succession candidates DataFrame for JSON response."""
    result = []
    for idx, row in candidates.head(max_count).iterrows():
        timeline = planner.estimate_development_timeline(row["不足スキル数"])
        result.append(
            {
                "member_code": row["メンバーコード"],
                "member_name": row["メンバー名"],
                "current_position": row.get("現在の役職", ""),
                "current_grade": row.get("現在の等級", ""),
                "readiness_score": round(row["準備度スコア"], 3),
                "skill_match_rate": round(row["スキルマッチ度"], 3),
                "owned_skills_count": int(row["保有スキル数"]),
                "missing_skills_count": int(row["不足スキル数"]),
                "estimated_timeline": timeline,
            }
        )
    return result


def _format_simulation_comparison(df) -> List[Dict]:
    """Format simulation comparison DataFrame for JSON response."""
    result = []
    for idx, row in df.iterrows():
        result.append(
            {
                "group": row.get("グループ", ""),
                "current_members": int(row.get("現在のメンバー数", 0)),
                "simulated_members": int(row.get("シミュレーション後のメンバー数", 0)),
                "current_avg_skills": round(row.get("現在の平均スキル数", 0), 1),
                "simulated_avg_skills": round(row.get("シミュレーション後の平均スキル数", 0), 1),
                "member_change": int(row.get("メンバー数変化", 0)),
                "skill_change": round(row.get("平均スキル数変化", 0), 1),
            }
        )
    return result
