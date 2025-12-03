"""
Role-based Career Dashboard API endpoints.

This module provides API endpoints for role-based skill analysis,
complementing the member-based approach in career_dashboard.py.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import pandas as pd

from backend.utils import session_manager, load_and_transform_session_data
from backend.api.career_dashboard import create_gantt_chart

router = APIRouter()
logger = logging.getLogger(__name__)


# =========================================================
# Request/Response Models
# =========================================================


class RoleSkillsRequest(BaseModel):
    """Request for getting role-based skill frequency"""

    session_id: str
    role_name: str
    min_frequency: float = 0.1  # Minimum skill frequency (default: 10%)


class RoleSkillInfo(BaseModel):
    """Skill information with frequency for a role"""

    skill_code: str
    skill_name: str
    category: str
    frequency: float  # 0.0 to 1.0
    member_count: int  # Number of members with this skill
    priority: str  # "必須" (>=50%), "推奨" (30-50%), "オプショナル" (10-30%)


class RoleGapAnalysisRequest(BaseModel):
    """Request for role-based gap analysis"""

    session_id: str
    model_id: str
    source_member_code: str
    target_role: str
    min_frequency: float = 0.1  # Minimum skill frequency threshold
    min_total_score: float = 0.3
    min_readiness_score: float = 0.0


class RoleCareerPathRequest(BaseModel):
    """Request for generating role-based career path"""

    session_id: str
    model_id: str
    source_member_code: str
    target_role: str
    min_frequency: float = 0.1  # Minimum skill frequency threshold
    min_total_score: float = 0.3
    min_readiness_score: float = 0.0
    min_effect_threshold: float = 0.03


class RoleRoadmapRequest(BaseModel):
    """Request for generating role-based career roadmap (Gantt chart)"""

    session_id: str
    model_id: str
    source_member_code: str
    target_role: str
    min_frequency: float = 0.1
    min_total_score: float = 0.3
    min_readiness_score: float = 0.0
    min_effect_threshold: float = 0.03


# =========================================================
# Helper Functions
# =========================================================


def get_member_info(members_df: pd.DataFrame, member_code: str) -> Dict[str, Any]:
    """Get member information from members dataframe"""
    member_row = members_df[members_df["メンバーコード"] == member_code]
    if len(member_row) == 0:
        return {"member_code": member_code, "member_name": member_code, "role": "不明"}

    row = member_row.iloc[0]
    return {
        "member_code": member_code,
        "member_name": row.get("メンバー名", member_code),
        "role": row.get("役職", "不明"),
    }


def get_member_skills_codes(member_competence_df: pd.DataFrame, member_code: str) -> List[str]:
    """Get skill codes for a member"""
    return member_competence_df[member_competence_df["メンバーコード"] == member_code][
        "力量コード"
    ].tolist()


def calculate_gap_skills(source_skills: List[str], target_skills: List[str]) -> List[str]:
    """Calculate missing skills (gap) between source and target"""
    source_set = set(source_skills)
    target_set = set(target_skills)
    gap = target_set - source_set
    return list(gap)


def calculate_role_skill_frequency(
    members_df: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    role_name: str,
    min_frequency: float = 0.1,
) -> Dict[str, Any]:
    """
    Calculate skill frequency for a specific role.

    Returns:
        Dictionary with role info and skill frequency data
    """
    # Get members with this role
    role_members = members_df[members_df["役職"] == role_name]
    total_members = len(role_members)

    if total_members == 0:
        return {"total_members": 0, "skills": {}, "skill_codes": []}

    role_member_codes = role_members["メンバーコード"].tolist()

    # Calculate skill frequency
    skill_frequency = {}
    skill_codes = []

    for skill_code in competence_master_df["力量コード"].unique():
        # Count how many members in this role have this skill
        count = len(
            member_competence_df[
                (member_competence_df["メンバーコード"].isin(role_member_codes))
                & (member_competence_df["力量コード"] == skill_code)
            ]
        )

        if count > 0:
            frequency = count / total_members
            if frequency >= min_frequency:
                skill_frequency[skill_code] = {"count": count, "frequency": frequency}
                skill_codes.append(skill_code)

    return {"total_members": total_members, "skills": skill_frequency, "skill_codes": skill_codes}


# =========================================================
# API Endpoints
# =========================================================


@router.post("/role-skills")
async def get_role_skills(request: RoleSkillsRequest):
    """
    Get skill frequency for a specific role.

    Args:
        request: Contains session_id, role_name, and min_frequency

    Returns:
        List of skills with frequency, member_count, and priority
    """
    try:
        data = load_and_transform_session_data(request.session_id)
        members_df = data["members_clean"]
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        if "役職" not in members_df.columns:
            raise HTTPException(status_code=400, detail="メンバーマスタに「役職」列が存在しません")

        # Calculate role skill frequency
        role_data = calculate_role_skill_frequency(
            members_df,
            member_competence_df,
            competence_master_df,
            request.role_name,
            request.min_frequency,
        )

        if role_data["total_members"] == 0:
            return {
                "success": False,
                "message": f"役職「{request.role_name}」のメンバーが見つかりません",
                "role_name": request.role_name,
                "total_members": 0,
                "skills": [],
            }

        # Get skill details and create response
        skills_list = []
        for skill_code, freq_data in role_data["skills"].items():
            skill_info = competence_master_df[competence_master_df["力量コード"] == skill_code]

            if len(skill_info) > 0:
                skill_row = skill_info.iloc[0]
                frequency = freq_data["frequency"]

                # Determine priority
                if frequency >= 0.5:
                    priority = "必須"
                elif frequency >= 0.3:
                    priority = "推奨"
                else:
                    priority = "オプショナル"

                skills_list.append(
                    {
                        "skill_code": skill_code,
                        "skill_name": skill_row.get("力量名", skill_code),
                        "category": skill_row.get("カテゴリー", "未分類"),
                        "frequency": frequency,
                        "member_count": freq_data["count"],
                        "priority": priority,
                    }
                )

        # Sort by frequency descending
        skills_list.sort(key=lambda x: x["frequency"], reverse=True)

        return {
            "success": True,
            "role_name": request.role_name,
            "total_members": role_data["total_members"],
            "skills": skills_list,
            "skill_count": len(skills_list),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ROLE_DASHBOARD] Error getting role skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gap-analysis")
async def analyze_role_gap(request: RoleGapAnalysisRequest):
    """
    Analyze skill gap between source member and target role.

    Args:
        request: Contains session_id, model_id, source member code, target role

    Returns:
        Gap analysis results with missing skills and frequency info
    """
    try:
        # Load data
        data = load_and_transform_session_data(request.session_id)
        members_df = data["members_clean"]
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        # Get source member info and skills
        source_info = get_member_info(members_df, request.source_member_code)
        source_codes = get_member_skills_codes(member_competence_df, request.source_member_code)

        # Calculate role skill frequency
        role_data = calculate_role_skill_frequency(
            members_df,
            member_competence_df,
            competence_master_df,
            request.target_role,
            request.min_frequency,
        )

        if role_data["total_members"] == 0:
            raise HTTPException(
                status_code=404, detail=f"役職「{request.target_role}」のメンバーが見つかりません"
            )

        target_codes = role_data["skill_codes"]
        total_members = role_data["total_members"]

        # Create target info for role
        target_info = {
            "member_code": f"役職:{request.target_role}",
            "member_name": f"{request.target_role}（{total_members}人の統合プロファイル）",
            "role": request.target_role,
        }

        # Calculate gap
        gap_codes = calculate_gap_skills(source_codes, target_codes)

        # Get gap skill details with frequency info
        gap_skills = []
        for code in gap_codes:
            skill_info = competence_master_df[competence_master_df["力量コード"] == code]
            if len(skill_info) > 0:
                skill_row = skill_info.iloc[0]
                freq_data = role_data["skills"].get(code, {})
                frequency = freq_data.get("frequency", 0)

                # Determine priority
                if frequency >= 0.5:
                    priority = "必須"
                elif frequency >= 0.3:
                    priority = "推奨"
                else:
                    priority = "オプショナル"

                gap_skills.append(
                    {
                        "skill_code": code,
                        "skill_name": skill_row.get("力量名", code),
                        "category": skill_row.get("カテゴリー", "未分類"),
                        "frequency": frequency,
                        "priority": priority,
                    }
                )

        return {
            "success": True,
            "source_member": source_info,
            "target_member": target_info,
            "gap_skills": gap_skills,
            "gap_count": len(gap_skills),
            "source_skill_count": len(source_codes),
            "target_skill_count": len(target_codes),
            "completion_rate": (
                (len(source_codes) / len(target_codes) * 100) if len(target_codes) > 0 else 100
            ),
            "is_role_based": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ROLE_DASHBOARD] Error in gap analysis: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/career-path")
async def generate_role_career_path(request: RoleCareerPathRequest):
    """
    Generate causal-filtered career learning path for a target role.

    Args:
        request: Contains session info, source member code, target role, and filtering parameters

    Returns:
        Recommended learning path with dependencies and roadmap data
    """
    try:
        # Get the trained model
        recommender = session_manager.get_model(request.model_id)
        if not recommender:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

        # Load data
        data = load_and_transform_session_data(request.session_id)
        members_df = data["members_clean"]
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        # Get source skills
        source_skills_codes = get_member_skills_codes(
            member_competence_df, request.source_member_code
        )

        # Calculate role skill frequency
        role_data = calculate_role_skill_frequency(
            members_df,
            member_competence_df,
            competence_master_df,
            request.target_role,
            request.min_frequency,
        )

        if role_data["total_members"] == 0:
            raise HTTPException(
                status_code=404, detail=f"役職「{request.target_role}」のメンバーが見つかりません"
            )

        target_skills_codes = role_data["skill_codes"]

        # Calculate gap
        gap_codes = calculate_gap_skills(source_skills_codes, target_skills_codes)

        # Get causal recommendations for the source member
        all_recommendations = recommender.recommend(
            request.source_member_code, top_n=100  # Get many to filter later
        )

        if not all_recommendations:
            return {
                "success": True,
                "recommended_skills": [],
                "message": "推薦スキルが見つかりませんでした",
            }

        # Filter recommendations to only include gap skills with good scores
        filtered_recommendations = []
        for rec in all_recommendations:
            # Get skill code from name
            skill_info = competence_master_df[
                competence_master_df["力量名"] == rec["competence_name"]
            ]

            if len(skill_info) == 0:
                continue

            skill_code = skill_info.iloc[0]["力量コード"]

            # Only include if it's in the gap
            if skill_code not in gap_codes:
                continue

            # Apply score filters
            if rec["score"] < request.min_total_score:
                continue

            details = rec.get("details", {})
            readiness = details.get("readiness_score_normalized", 0)

            if readiness < request.min_readiness_score:
                continue

            filtered_recommendations.append(
                {
                    "competence_code": skill_code,
                    "competence_name": rec["competence_name"],
                    "category": skill_info.iloc[0].get("カテゴリー", "未分類"),
                    "total_score": rec["score"],
                    "readiness_score": readiness,
                    "bayesian_score": details.get("bayesian_score_normalized", 0),
                    "utility_score": details.get("utility_score_normalized", 0),
                    "readiness_reasons": details.get("readiness_reasons", [])[:3],
                    "utility_reasons": details.get("utility_reasons", [])[:3],
                    "explanation": rec.get("explanation", ""),
                }
            )

        # Calculate dependencies using causal adjacency matrix
        adj_matrix = recommender.learner.get_adjacency_matrix()

        # Add dependency information
        for skill in filtered_recommendations:
            skill_name = skill["competence_name"]

            prerequisites = []
            enables = []

            if skill_name in adj_matrix.index and skill_name in adj_matrix.columns:
                # Prerequisites: skills that causally affect this skill
                incoming = adj_matrix[skill_name]
                for other_skill, effect in incoming.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        # Check if this prerequisite is in our filtered list
                        if any(
                            s["competence_name"] == other_skill for s in filtered_recommendations
                        ):
                            prerequisites.append(
                                {"skill_name": other_skill, "effect": float(effect)}
                            )

                # Enables: skills that this skill causally affects
                outgoing = adj_matrix.loc[skill_name]
                for other_skill, effect in outgoing.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        if any(
                            s["competence_name"] == other_skill for s in filtered_recommendations
                        ):
                            enables.append({"skill_name": other_skill, "effect": float(effect)})

            skill["prerequisites"] = prerequisites
            skill["enables"] = enables

        # Calculate estimated timeline
        total_deps = sum(len(s["prerequisites"]) for s in filtered_recommendations)
        estimated_weeks = len(filtered_recommendations) * 2 + total_deps
        estimated_months = estimated_weeks / 4

        return {
            "success": True,
            "recommended_skills": filtered_recommendations,
            "skill_count": len(filtered_recommendations),
            "avg_score": (
                sum(s["total_score"] for s in filtered_recommendations)
                / len(filtered_recommendations)
                if filtered_recommendations
                else 0
            ),
            "total_dependencies": total_deps,
            "estimated_months": round(estimated_months, 1),
            "message": f"{len(filtered_recommendations)}個のスキルを推薦しました",
            "is_role_based": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ROLE_DASHBOARD] Error generating career path: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/career-roadmap")
async def generate_role_career_roadmap(request: RoleRoadmapRequest):
    """
    Generate role-based career roadmap Gantt chart.

    Args:
        request: Contains session info, source member, target role, and filtering parameters

    Returns:
        Plotly Gantt chart as JSON
    """
    try:
        # Get the trained model
        recommender = session_manager.get_model(request.model_id)
        if not recommender:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

        # Load data
        data = load_and_transform_session_data(request.session_id)
        members_df = data["members_clean"]
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        # Get source skills
        source_skills_codes = get_member_skills_codes(
            member_competence_df, request.source_member_code
        )

        # Calculate role skill frequency
        role_data = calculate_role_skill_frequency(
            members_df,
            member_competence_df,
            competence_master_df,
            request.target_role,
            request.min_frequency,
        )

        if role_data["total_members"] == 0:
            raise HTTPException(
                status_code=404, detail=f"役職「{request.target_role}」のメンバーが見つかりません"
            )

        target_skills_codes = role_data["skill_codes"]

        # Calculate gap
        gap_codes = calculate_gap_skills(source_skills_codes, target_skills_codes)

        # Get causal recommendations for the source member
        all_recommendations = recommender.recommend(
            request.source_member_code, top_n=100  # Get many to filter later
        )

        if not all_recommendations:
            return {
                "success": True,
                "gantt_chart": {},
                "message": "推薦スキルが見つかりませんでした",
            }

        # Filter recommendations to only include gap skills with good scores
        filtered_recommendations = []
        for rec in all_recommendations:
            # Get skill code from name
            skill_info = competence_master_df[
                competence_master_df["力量名"] == rec["competence_name"]
            ]

            if len(skill_info) == 0:
                continue

            skill_code = skill_info.iloc[0]["力量コード"]

            # Only include if it's in the gap
            if skill_code not in gap_codes:
                continue

            # Apply score filters
            if rec["score"] < request.min_total_score:
                continue

            details = rec.get("details", {})
            readiness = details.get("readiness_score_normalized", 0)

            if readiness < request.min_readiness_score:
                continue

            filtered_recommendations.append(
                {
                    "competence_code": skill_code,
                    "competence_name": rec["competence_name"],
                    "category": skill_info.iloc[0].get("カテゴリー", "未分類"),
                    "total_score": rec["score"],
                    "readiness_score": readiness,
                    "bayesian_score": details.get("bayesian_score_normalized", 0),
                    "utility_score": details.get("utility_score_normalized", 0),
                }
            )

        # Calculate dependencies using causal adjacency matrix
        adj_matrix = recommender.learner.get_adjacency_matrix()

        # Build dependencies dict
        dependencies = {}

        for skill in filtered_recommendations:
            skill_name = skill["competence_name"]
            skill_code = skill["competence_code"]

            prerequisites = []
            enables = []

            if skill_name in adj_matrix.index and skill_name in adj_matrix.columns:
                # Prerequisites
                incoming = adj_matrix[skill_name]
                for other_skill, effect in incoming.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        if any(
                            s["competence_name"] == other_skill for s in filtered_recommendations
                        ):
                            prerequisites.append(
                                {"skill_name": other_skill, "effect": float(effect)}
                            )

                # Enables
                outgoing = adj_matrix.loc[skill_name]
                for other_skill, effect in outgoing.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        if any(
                            s["competence_name"] == other_skill for s in filtered_recommendations
                        ):
                            enables.append({"skill_name": other_skill, "effect": float(effect)})

            dependencies[skill_code] = {"prerequisites": prerequisites, "enables": enables}

        # Create Gantt chart
        target_name = f"役職: {request.target_role}"
        fig_dict = create_gantt_chart(filtered_recommendations, dependencies, target_name)

        return {"success": True, "gantt_chart": fig_dict, "message": "ロードマップを生成しました"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ROLE_DASHBOARD] Error generating career roadmap: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
