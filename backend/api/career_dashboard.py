"""
Career Dashboard API endpoints for employee career development analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

from backend.utils import session_manager, load_and_transform_session_data

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


class RoadmapRequest(BaseModel):
    """Request for generating career roadmap (Gantt chart)"""

    session_id: str
    model_id: str
    source_member_code: str
    target_member_code: str
    target_member_name: str
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
    member_row = members_df[members_df["メンバーコード"] == member_code]
    if len(member_row) == 0:
        return {"member_code": member_code, "member_name": member_code, "role": "不明"}

    row = member_row.iloc[0]
    return {
        "member_code": member_code,
        "member_name": row.get("メンバー名", member_code),
        "role": row.get("役職", "不明"),
    }


def get_member_skills(
    member_competence_df: pd.DataFrame, competence_master_df: pd.DataFrame, member_code: str
) -> List[Dict[str, Any]]:
    """Get all skills for a member with details"""
    member_skills = member_competence_df[member_competence_df["メンバーコード"] == member_code]

    skills = []
    for _, row in member_skills.iterrows():
        skill_code = row["力量コード"]

        # Get skill details from master
        skill_info = competence_master_df[competence_master_df["力量コード"] == skill_code]

        if len(skill_info) > 0:
            skill_row = skill_info.iloc[0]
            skills.append(
                {
                    "skill_code": skill_code,
                    "skill_name": skill_row.get("力量名", skill_code),
                    "category": skill_row.get("カテゴリー", "未分類"),
                    "level": 1,  # Default level
                }
            )

    return skills


def calculate_gap_skills(source_skills: List[str], target_skills: List[str]) -> List[str]:
    """Calculate missing skills (gap) between source and target"""
    source_set = set(source_skills)
    target_set = set(target_skills)
    gap = target_set - source_set
    return list(gap)


def topological_sort(skills: List[Dict], dependencies: Dict[str, Dict]) -> List[List[str]]:
    """
    Topological sort to determine learning order.

    Returns:
        [[layer1_skills], [layer2_skills], ...]
        Skills in the same layer can be learned in parallel
    """
    # Build graph structure
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    all_skills = {s["competence_code"]: s for s in skills}
    skill_codes = set(all_skills.keys())

    for skill in skills:
        skill_code = skill["competence_code"]
        prereqs = dependencies.get(skill_code, {}).get("prerequisites", [])
        in_degree[skill_code] = len(prereqs)

        enables = dependencies.get(skill_code, {}).get("enables", [])
        for next_skill_info in enables:
            next_skill_code = next_skill_info.get("skill_code")
            if next_skill_code and next_skill_code in skill_codes:
                graph[skill_code].append(next_skill_code)

    # Kahn's algorithm for topological sort
    layers = []
    current_layer = [s for s in skill_codes if in_degree[s] == 0]

    while current_layer:
        layers.append(sorted(current_layer))  # Stable sort
        next_layer = []

        for skill in current_layer:
            for next_skill in graph[skill]:
                in_degree[next_skill] -= 1
                if in_degree[next_skill] == 0:
                    next_layer.append(next_skill)

        current_layer = next_layer

    logger.info(f"Topological sort complete: {len(layers)} layers")

    return layers


def estimate_duration(skill_code: str, dependencies: Dict[str, Dict]) -> int:
    """
    Estimate duration for skill acquisition (in weeks).

    Base: 2 weeks
    More prerequisites: +prerequisite count
    """
    base_duration = 2
    prereq_count = len(dependencies.get(skill_code, {}).get("prerequisites", []))
    return base_duration + min(prereq_count, 4)  # Max 6 weeks


def calculate_schedule(
    learning_order: List[List[str]], skills: List[Dict], dependencies: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Calculate start and end times for each skill.

    Returns:
        {
            skill_code: {
                "start_week": int,
                "end_week": int,
                "duration_weeks": int
            }
        }
    """
    schedule = {}
    current_week = 0
    skill_map = {s["competence_code"]: s for s in skills}

    for layer_idx, layer in enumerate(learning_order):
        # Calculate max duration in this layer
        max_duration = 0
        for skill_code in layer:
            duration = estimate_duration(skill_code, dependencies)
            max_duration = max(max_duration, duration)

        # Set same start time for all skills in layer (parallel learning)
        for skill_code in layer:
            duration = estimate_duration(skill_code, dependencies)

            schedule[skill_code] = {
                "start_week": current_week,
                "end_week": current_week + duration,
                "duration_weeks": duration,
            }

        # Next layer starts after longest skill in current layer completes
        current_week += max_duration

    return schedule


def create_gantt_chart(
    skills: List[Dict], dependencies: Dict[str, Dict], target_member_name: str = "未設定"
) -> Dict:
    """
    Create Gantt chart based on dependencies.

    Args:
        skills: List of recommended skills with causal scores
        dependencies: Dependency information
        target_member_name: Target member name

    Returns:
        Plotly figure as JSON
    """
    if not skills:
        logger.warning("Skills list is empty")
        return {}

    # Topological sort to determine learning order
    learning_order = topological_sort(skills, dependencies)

    # Calculate schedule for each skill
    schedule = calculate_schedule(learning_order, skills, dependencies)

    # Create data structure for Plotly Gantt chart
    tasks = []
    colors = []

    skill_map = {s["competence_code"]: s for s in skills}

    for skill_code, timing in schedule.items():
        skill = skill_map.get(skill_code)
        if not skill:
            continue

        task_dict = {
            "Task": skill["competence_name"],
            "Start": timing["start_week"],
            "Finish": timing["end_week"],
            "Duration": timing["duration_weeks"],
            "Resource": f"スコア: {skill['total_score']:.2f}",
        }
        tasks.append(task_dict)

        # Color by score
        if skill["total_score"] >= 0.7:
            colors.append("#2ecc71")  # High priority: green
        elif skill["total_score"] >= 0.4:
            colors.append("#3498db")  # Medium priority: blue
        else:
            colors.append("#95a5a6")  # Low priority: gray

    # Sort by start time in reverse (Plotly draws from bottom to top)
    sorted_indices = sorted(range(len(tasks)), key=lambda i: tasks[i]["Start"], reverse=True)
    tasks = [tasks[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    # Create Plotly Gantt chart
    fig = go.Figure()

    for i, task in enumerate(tasks):
        fig.add_trace(
            go.Bar(
                x=[task["Duration"]],
                y=[task["Task"]],
                base=[task["Start"]],
                orientation="h",
                marker=dict(color=colors[i]),
                name=task["Resource"],
                text=task["Resource"],
                textposition="inside",
                hovertemplate=(
                    f"<b>{task['Task']}</b><br>"
                    f"{task['Resource']}<br>"
                    f"開始: Week {task['Start']}<br>"
                    f"完了予定: Week {task['Finish']}<br>"
                    f"期間: {task['Duration']}週間<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"📅 キャリアロードマップ - {target_member_name}",
        xaxis_title="週",
        yaxis_title="スキル",
        height=max(400, len(tasks) * 40),
        showlegend=False,
        barmode="overlay",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )

    return fig.to_dict()


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
        members_df = data["members_clean"]
        member_competence_df = data["member_competence"]

        members_list = []
        for _, row in members_df.iterrows():
            member_code = str(row.get("メンバーコード", ""))
            if not member_code:
                continue

            # Count skills for this member
            skill_count = len(
                member_competence_df[member_competence_df["メンバーコード"] == member_code]
            )

            members_list.append(
                {
                    "member_code": member_code,
                    "member_name": str(row.get("メンバー名", "")),
                    "role": str(row.get("役職", "未設定")),
                    "skill_count": skill_count,
                    "display_name": f"{member_code} - {row.get('メンバー名', '')} ({row.get('役職', '未設定')})",
                }
            )

        # Sort by member code
        members_list.sort(key=lambda x: x["member_code"])

        return {"success": True, "members": members_list, "total_count": len(members_list)}

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
        members_df = data["members_clean"]

        if "役職" not in members_df.columns:
            return {"success": False, "message": "役職列が見つかりません", "roles": []}

        # Get role counts
        role_counts = members_df["役職"].value_counts().to_dict()

        roles_list = []
        for role, count in role_counts.items():
            if pd.isna(role) or role == "":
                continue

            roles_list.append({"role_name": str(role), "member_count": int(count)})

        # Sort by member count descending
        roles_list.sort(key=lambda x: x["member_count"], reverse=True)

        return {"success": True, "roles": roles_list, "total_count": len(roles_list)}

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
        members_df = data["members_clean"]
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        # Get member info
        member_info = get_member_info(members_df, request.member_code)

        # Get skills
        skills = get_member_skills(member_competence_df, competence_master_df, request.member_code)

        return {
            "success": True,
            "member_info": member_info,
            "current_skills": skills,
            "skill_count": len(skills),
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
        members_df = data["members_clean"]
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        # Get member info
        source_info = get_member_info(members_df, request.source_member_code)
        target_info = get_member_info(members_df, request.target_member_code)

        # Get skills for both members
        source_skills_list = get_member_skills(
            member_competence_df, competence_master_df, request.source_member_code
        )

        target_skills_list = get_member_skills(
            member_competence_df, competence_master_df, request.target_member_code
        )

        # Calculate gap
        source_codes = [s["skill_code"] for s in source_skills_list]
        target_codes = [s["skill_code"] for s in target_skills_list]
        gap_codes = calculate_gap_skills(source_codes, target_codes)

        # Get gap skill details
        gap_skills = []
        for code in gap_codes:
            skill_info = competence_master_df[competence_master_df["力量コード"] == code]
            if len(skill_info) > 0:
                skill_row = skill_info.iloc[0]
                gap_skills.append(
                    {
                        "skill_code": code,
                        "skill_name": skill_row.get("力量名", code),
                        "category": skill_row.get("カテゴリー", "未分類"),
                    }
                )

        return {
            "success": True,
            "source_member": source_info,
            "target_member": target_info,
            "gap_skills": gap_skills,
            "gap_count": len(gap_skills),
            "source_skill_count": len(source_skills_list),
            "target_skill_count": len(target_skills_list),
            "completion_rate": (
                (len(source_codes) / len(target_codes) * 100) if len(target_codes) > 0 else 100
            ),
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
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

        # Load data
        data = load_and_transform_session_data(request.session_id)
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        # Get gap analysis first
        source_skills_codes = member_competence_df[
            member_competence_df["メンバーコード"] == request.source_member_code
        ]["力量コード"].tolist()

        target_skills_codes = member_competence_df[
            member_competence_df["メンバーコード"] == request.target_member_code
        ]["力量コード"].tolist()

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
                competence_master_df["力量名"] == rec.get("skill_name", rec.get("competence_name", ""))
            ]

            if len(skill_info) == 0:
                continue

            skill_code = skill_info.iloc[0]["力量コード"]

            # Only include if it's in the gap
            if skill_code not in gap_codes:
                continue

            # Apply score filters
            if rec.get("total_score", rec.get("score", 0)) < request.min_total_score:
                continue

            details = rec.get("details", {})
            readiness = details.get("readiness_score_normalized", 0)

            if readiness < request.min_readiness_score:
                continue

            skill_name = rec.get("skill_name", rec.get("competence_name", ""))
            filtered_recommendations.append(
                {
                    "competence_code": skill_code,
                    "competence_name": skill_name,
                    "category": skill_info.iloc[0].get("カテゴリー", "未分類"),
                    "total_score": rec.get("total_score", rec.get("score", 0)),
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
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] Error generating career path: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/career-roadmap")
async def generate_career_roadmap(request: RoadmapRequest):
    """
    Generate career roadmap Gantt chart.

    Args:
        request: Contains session info, member codes, target name, and filtering parameters

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
        member_competence_df = data["member_competence"]
        competence_master_df = data["competence_master"]

        # Get gap analysis first
        source_skills_codes = member_competence_df[
            member_competence_df["メンバーコード"] == request.source_member_code
        ]["力量コード"].tolist()

        target_skills_codes = member_competence_df[
            member_competence_df["メンバーコード"] == request.target_member_code
        ]["力量コード"].tolist()

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
                competence_master_df["力量名"] == rec.get("skill_name", rec.get("competence_name", ""))
            ]

            if len(skill_info) == 0:
                continue

            skill_code = skill_info.iloc[0]["力量コード"]

            # Only include if it's in the gap
            if skill_code not in gap_codes:
                continue

            # Apply score filters
            if rec.get("total_score", rec.get("score", 0)) < request.min_total_score:
                continue

            details = rec.get("details", {})
            readiness = details.get("readiness_score_normalized", 0)

            if readiness < request.min_readiness_score:
                continue

            skill_name = rec.get("skill_name", rec.get("competence_name", ""))
            filtered_recommendations.append(
                {
                    "competence_code": skill_code,
                    "competence_name": skill_name,
                    "category": skill_info.iloc[0].get("カテゴリー", "未分類"),
                    "total_score": rec.get("total_score", rec.get("score", 0)),
                    "readiness_score": readiness,
                    "bayesian_score": details.get("bayesian_score_normalized", 0),
                    "utility_score": details.get("utility_score_normalized", 0),
                }
            )

        # Calculate dependencies using causal adjacency matrix
        adj_matrix = recommender.learner.get_adjacency_matrix()

        # Build dependencies dict
        dependencies = {}
        code_to_name = dict(zip(competence_master_df["力量コード"], competence_master_df["力量名"]))
        name_to_code = {v: k for k, v in code_to_name.items()}

        for skill in filtered_recommendations:
            skill_code = skill["competence_code"]
            skill_name = skill["competence_name"]

            prerequisites = []
            enables = []

            if skill_name in adj_matrix.index and skill_name in adj_matrix.columns:
                # Prerequisites: skills that causally affect this skill
                incoming = adj_matrix[skill_name]
                for other_skill, effect in incoming.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        other_code = name_to_code.get(other_skill)
                        # Check if this prerequisite is in our filtered list
                        if other_code and any(
                            s["competence_code"] == other_code for s in filtered_recommendations
                        ):
                            prerequisites.append(
                                {
                                    "skill_name": other_skill,
                                    "skill_code": other_code,
                                    "effect": float(effect),
                                }
                            )

                # Enables: skills that this skill causally affects
                outgoing = adj_matrix.loc[skill_name]
                for other_skill, effect in outgoing.items():
                    if abs(effect) >= request.min_effect_threshold and effect > 0:
                        other_code = name_to_code.get(other_skill)
                        if other_code and any(
                            s["competence_code"] == other_code for s in filtered_recommendations
                        ):
                            enables.append(
                                {
                                    "skill_name": other_skill,
                                    "skill_code": other_code,
                                    "effect": float(effect),
                                }
                            )

            dependencies[skill_code] = {
                "prerequisites": prerequisites,
                "enables": enables,
            }

        # Create Gantt chart
        gantt_chart = create_gantt_chart(
            filtered_recommendations, dependencies, request.target_member_name
        )

        return {
            "success": True,
            "gantt_chart": gantt_chart,
            "skill_count": len(filtered_recommendations),
            "message": f"{len(filtered_recommendations)}個のスキルでロードマップを作成しました",
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] Error generating career roadmap: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
