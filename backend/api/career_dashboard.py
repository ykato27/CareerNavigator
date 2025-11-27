"""
Career Dashboard API endpoints for employee career development analysis.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import pandas as pd

from backend.utils import session_manager, load_csv_files, load_and_transform_session_data, clean_dataframe_columns

router = APIRouter()
logger = logging.getLogger(__name__)


# =========================================================
# Request/Response Models
# =========================================================

class MemberSkillsRequest(BaseModel):
    """Request for getting member's current skills"""
    session_id: str
    member_code: str


class GapAnalysisRequest(BaseModel):
    """Request for career gap analysis"""
    session_id: str
    model_id: str
    source_member_code: str
    target_member_code: str
    min_total_score: float = 0.3
    min_readiness_score: float = 0.0


class CareerPathRequest(BaseModel):
    """Request for generating career path with dependencies"""
    session_id: str
    model_id: str
    source_member_code: str
    target_member_code: str
    min_total_score: float = 0.3
    min_readiness_score: float = 0.0
    min_effect_threshold: float = 0.03


class SkillInfo(BaseModel):
    """Individual skill information"""
    skill_code: str
    skill_name: str
    category: str
    level: int = 1


class RecommendedSkill(BaseModel):
    """Recommended skill with causal scores"""
    competence_code: str
    competence_name: str
    category: str
    total_score: float
    readiness_score: float
    bayesian_score: float
    utility_score: float
    readiness_reasons: List[tuple]
    utility_reasons: List[tuple]
    prerequisites: List[str]
    enables: List[str]


# =========================================================
# Helper Functions
# =========================================================

def get_member_info(members_df: pd.DataFrame, member_code: str) -> Dict[str, Any]:
    """Get member information from members dataframe"""
    member_row = members_df[members_df['メンバーコード'] == member_code]
    if len(member_row) == 0:
        return {
            "member_code": member_code,
            "member_name": member_code,
            "role": "不明"
        }

    row = member_row.iloc[0]
    return {
        "member_code": member_code,
        "member_name": row.get('メンバー名', member_code),
        "role": row.get('役職', '不明')
    }


def get_member_skills(member_competence_df: pd.DataFrame,
                     competence_master_df: pd.DataFrame,
                     member_code: str) -> List[Dict[str, Any]]:
    """Get all skills for a member with details"""
    member_skills = member_competence_df[
        member_competence_df['メンバーコード'] == member_code
    ]

    skills = []
    for _, row in member_skills.iterrows():
        skill_code = row['力量コード']

        # Get skill details from master
        skill_info = competence_master_df[
            competence_master_df['力量コード'] == skill_code
        ]

        if len(skill_info) > 0:
            skill_row = skill_info.iloc[0]
            skills.append({
                "skill_code": skill_code,
                "skill_name": skill_row.get('力量名', skill_code),
                "category": skill_row.get('カテゴリー', '未分類'),
                "level": 1  # Default level
            })

    return skills


def calculate_gap_skills(source_skills: List[str],
                        target_skills: List[str]) -> List[str]:
    """Calculate missing skills (gap) between source and target"""
    source_set = set(source_skills)
    target_set = set(target_skills)
    gap = target_set - source_set
    return list(gap)


# =========================================================
# API Endpoints
# =========================================================

@router.get("/members")
async def get_available_members(session_id: str):
    """
    Get list of all available members for selection.

    Args:
        session_id: Session ID for loaded data

    Returns:
        List of members with code, name, role, and skill count
    """
    try:
        data = load_and_transform_session_data(session_id)
        members_df = data['members_clean']
        member_competence_df = data['member_competence']

        members_list = []
        for _, row in members_df.iterrows():
            member_code = str(row.get('メンバーコード', ''))
            if not member_code:
                continue

            # Count skills for this member
            skill_count = len(member_competence_df[
                member_competence_df['メンバーコード'] == member_code
            ])

            members_list.append({
                "member_code": member_code,
                "member_name": str(row.get('メンバー名', '')),
                "role": str(row.get('役職', '未設定')),
                "skill_count": skill_count,
                "display_name": f"{member_code} - {row.get('メンバー名', '')} ({row.get('役職', '未設定')})"
            })

        # Sort by member code
        members_list.sort(key=lambda x: x['member_code'])

        return {
            "success": True,
            "members": members_list,
            "total_count": len(members_list)
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] Error getting members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roles")
async def get_available_roles(session_id: str):
    """
    Get list of all available roles/positions.

    Args:
        session_id: Session ID for loaded data

    Returns:
        List of roles with member counts
    """
    try:
        data = load_and_transform_session_data(session_id)
        members_df = data['members_clean']

        if '役職' not in members_df.columns:
            return {
                "success": False,
                "message": "役職列が見つかりません",
                "roles": []
            }

        # Get role counts
        role_counts = members_df['役職'].value_counts().to_dict()

        roles_list = []
        for role, count in role_counts.items():
            if pd.isna(role) or role == '':
                continue

            roles_list.append({
                "role_name": str(role),
                "member_count": int(count)
            })

        # Sort by member count descending
        roles_list.sort(key=lambda x: x['member_count'], reverse=True)

        return {
            "success": True,
            "roles": roles_list,
            "total_count": len(roles_list)
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] Error getting roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/member-skills")
async def get_member_current_skills(request: MemberSkillsRequest):
    """
    Get current skills for a specific member.

    Args:
        request: Contains session_id and member_code

    Returns:
        Member info and list of current skills
    """
    try:
        data = load_and_transform_session_data(request.session_id)
        members_df = data['members_clean']
        member_competence_df = data['member_competence']
        competence_master_df = data['competence_master']

        # Get member info
        member_info = get_member_info(members_df, request.member_code)

        # Get skills
        skills = get_member_skills(
            member_competence_df,
            competence_master_df,
            request.member_code
        )

        return {
            "success": True,
            "member_info": member_info,
            "current_skills": skills,
            "skill_count": len(skills)
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] Error getting member skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gap-analysis")
async def analyze_career_gap(request: GapAnalysisRequest):
    """
    Analyze skill gap between source member and target member.

    Args:
        request: Contains session_id, model_id, source/target member codes, and filters

    Returns:
        Gap analysis results with missing skills
    """
    try:
        # Load data
        data = load_and_transform_session_data(request.session_id)
        members_df = data['members_clean']
        member_competence_df = data['member_competence']
        competence_master_df = data['competence_master']

        # Get member info
        source_info = get_member_info(members_df, request.source_member_code)
        target_info = get_member_info(members_df, request.target_member_code)

        # Get skills for both members
        source_skills_list = get_member_skills(
            member_competence_df,
            competence_master_df,
            request.source_member_code
        )

        target_skills_list = get_member_skills(
            member_competence_df,
            competence_master_df,
            request.target_member_code
        )

        # Calculate gap
        source_codes = [s['skill_code'] for s in source_skills_list]
        target_codes = [s['skill_code'] for s in target_skills_list]
        gap_codes = calculate_gap_skills(source_codes, target_codes)

        # Get gap skill details
        gap_skills = []
        for code in gap_codes:
            skill_info = competence_master_df[
                competence_master_df['力量コード'] == code
            ]
            if len(skill_info) > 0:
                skill_row = skill_info.iloc[0]
                gap_skills.append({
                    "skill_code": code,
                    "skill_name": skill_row.get('力量名', code),
                    "category": skill_row.get('カテゴリー', '未分類')
                })

        return {
            "success": True,
            "source_member": source_info,
            "target_member": target_info,
            "gap_skills": gap_skills,
            "gap_count": len(gap_skills),
            "source_skill_count": len(source_skills_list),
            "target_skill_count": len(target_skills_list),
            "completion_rate": (len(source_codes) / len(target_codes) * 100) if len(target_codes) > 0 else 100
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] Error in gap analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/career-path")
async def generate_career_path(request: CareerPathRequest):
    """
    Generate causal-filtered career learning path with dependencies.

    Args:
        request: Contains session info, member codes, and filtering parameters

    Returns:
        Recommended learning path with dependencies and roadmap data
    """
    try:
        # Get the trained model
        recommender = session_manager.get_model(request.model_id)
        if not recommender:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_id}' not found"
            )

        # Load data
        data = load_and_transform_session_data(request.session_id)
        member_competence_df = data['member_competence']
        competence_master_df = data['competence_master']

        # Get gap analysis first
        source_skills_codes = member_competence_df[
            member_competence_df['メンバーコード'] == request.source_member_code
        ]['力量コード'].tolist()

        target_skills_codes = member_competence_df[
            member_competence_df['メンバーコード'] == request.target_member_code
        ]['力量コード'].tolist()

        gap_codes = calculate_gap_skills(source_skills_codes, target_skills_codes)

        # Get causal recommendations for the source member
        all_recommendations = recommender.recommend(
            request.source_member_code,
            top_n=100  # Get many to filter later
        )

        if not all_recommendations:
            return {
                "success": True,
                "recommended_skills": [],
                "message": "推薦スキルが見つかりませんでした"
            }

        # Filter recommendations to only include gap skills with good scores
        filtered_recommendations = []
        for rec in all_recommendations:
            # Get skill code from name
            skill_info = competence_master_df[
                competence_master_df['力量名'] == rec['competence_name']
            ]

            if len(skill_info) == 0:
                continue

            skill_code = skill_info.iloc[0]['力量コード']

            # Only include if it's in the gap
            if skill_code not in gap_codes:
                continue

            # Apply score filters
            if rec['score'] < request.min_total_score:
                continue

            details = rec.get('details', {})
            readiness = details.get('readiness_score_normalized', 0)

            if readiness < request.min_readiness_score:
                continue

            filtered_recommendations.append({
                "competence_code": skill_code,
                "competence_name": rec['competence_name'],
                "category": skill_info.iloc[0].get('カテゴリー', '未分類'),
                "total_score": rec['score'],
                "readiness_score": readiness,
                "bayesian_score": details.get('bayesian_score_normalized', 0),
                "utility_score": details.get('utility_score_normalized', 0),
                "readiness_reasons": details.get('readiness_reasons', [])[:3],
                "utility_reasons": details.get('utility_reasons', [])[:3],
                "explanation": rec.get('explanation', '')
            })

        # Calculate dependencies using causal adjacency matrix
        adj_matrix = recommender.learner.get_adjacency_matrix()

        # Add dependency information
        for skill in filtered_recommendations:
            skill_name = skill['competence_name']

            prerequisites = []
            enables = []

            if skill_name in adj_matrix.index and skill_name in adj_matrix.columns:
                # Prerequisites: skills that causally affect this skill
                incoming = adj_matrix[skill_name]
                for other_skill, effect in incoming.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        # Check if this prerequisite is in our filtered list
                        if any(s['competence_name'] == other_skill for s in filtered_recommendations):
                            prerequisites.append({
                                "skill_name": other_skill,
                                "effect": float(effect)
                            })

                # Enables: skills that this skill causally affects
                outgoing = adj_matrix.loc[skill_name]
                for other_skill, effect in outgoing.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        if any(s['competence_name'] == other_skill for s in filtered_recommendations):
                            enables.append({
                                "skill_name": other_skill,
                                "effect": float(effect)
                            })

            skill['prerequisites'] = prerequisites
            skill['enables'] = enables

        # Calculate estimated timeline
        total_deps = sum(len(s['prerequisites']) for s in filtered_recommendations)
        estimated_weeks = len(filtered_recommendations) * 2 + total_deps
        estimated_months = estimated_weeks / 4

        return {
            "success": True,
            "recommended_skills": filtered_recommendations,
            "skill_count": len(filtered_recommendations),
            "avg_score": sum(s['total_score'] for s in filtered_recommendations) / len(filtered_recommendations) if filtered_recommendations else 0,
            "total_dependencies": total_deps,
            "estimated_months": round(estimated_months, 1),
            "message": f"{len(filtered_recommendations)}個のスキルを推薦しました"
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] Error generating career path: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
