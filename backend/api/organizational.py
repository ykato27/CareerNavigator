from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any, List
import pandas as pd
from pathlib import Path

from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.organizational.skill_gap_analyzer import SkillGapAnalyzer
from skillnote_recommendation.organizational import org_metrics
from skillnote_recommendation.strategic.succession_planner import SuccessionPlanner
from skillnote_recommendation.strategic.org_simulator import OrganizationSimulator

router = APIRouter()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# In-memory storage for session data
session_data_cache: Dict[str, Dict[str, Any]] = {}


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


def load_session_data(session_id: str) -> Dict[str, pd.DataFrame]:
    """Load session data from CSV files"""
    if session_id in session_data_cache:
        return session_data_cache[session_id]

    base_upload_dir = PROJECT_ROOT / "backend" / "temp_uploads"
    session_dir = base_upload_dir / session_id

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session data not found. Please upload data first.")

    # Load CSV files
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

    # Transform data
    transformer = DataTransformer()
    competence_master = transformer.create_competence_master(data)
    member_competence, valid_members = transformer.create_member_competence(data, competence_master)

    # Create members_clean DataFrame
    members_df = data['members'].copy()
    members_df = members_df[members_df['メンバーコード'].isin(valid_members)]

    transformed_data = {
        "member_competence": member_competence,
        "competence_master": competence_master,
        "members_clean": members_df
    }

    # Cache the transformed data
    session_data_cache[session_id] = transformed_data

    return transformed_data


@router.post("/organizational/metrics")
async def get_organizational_metrics(request: OrganizationalMetricsRequest):
    """
    Get organizational skill map metrics and dashboard data.
    """
    try:
        transformed_data = load_session_data(request.session_id)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]
        members_df = transformed_data["members_clean"]

        # Calculate KPI metrics
        total_members = len(members_df)
        total_skills = len(competence_master_df)
        total_skill_records = len(member_competence_df)
        avg_skills_per_member = total_skill_records / total_members if total_members > 0 else 0

        # Calculate coverage
        coverage_info = org_metrics.calculate_skill_coverage(
            member_competence_df, competence_master_df
        )

        # Calculate concentration
        concentration_info = org_metrics.calculate_skill_concentration(
            member_competence_df, threshold=3
        )

        # Calculate diversity
        diversity_index = org_metrics.calculate_skill_diversity_index(
            member_competence_df
        )

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
                top_skills.append({
                    "skill_code": skill_code,
                    "skill_name": skill_info.iloc[0]["力量名"],
                    "member_count": int(count)
                })

        return {
            "success": True,
            "metrics": {
                "total_members": total_members,
                "total_skills": total_skills,
                "avg_skills_per_member": round(avg_skills_per_member, 1),
                "coverage_rate": round(coverage_info.get('coverage_rate', 0), 3),
                "diversity_index": round(diversity_index, 2),
                "high_concentration_skills": concentration_info.get('high_concentration_skills', 0),
                "low_concentration_skills": concentration_info.get('low_concentration_skills', 0)
            },
            "category_distribution": category_distribution,
            "top_skills": top_skills
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get organizational metrics: {str(e)}")


@router.post("/organizational/skill-gap")
async def analyze_skill_gap(request: SkillGapAnalysisRequest):
    """
    Analyze skill gaps using top percentile approach.
    """
    try:
        transformed_data = load_session_data(request.session_id)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]

        # Initialize analyzer
        analyzer = SkillGapAnalyzer()

        # Calculate current profile
        current_profile = analyzer.calculate_current_profile(
            member_competence_df,
            competence_master_df
        )

        # Calculate target profile (top N% approach)
        target_profile = analyzer.calculate_target_profile_top_percentile(
            member_competence_df,
            competence_master_df,
            percentile=request.percentile
        )

        # Calculate gap
        gap_df = analyzer.calculate_gap(current_profile, target_profile)

        # Identify critical skills (gap rate > 30%)
        critical_skills = analyzer.identify_critical_skills(gap_df, threshold=0.3)

        # Format gap data for frontend
        gap_data = []
        for idx, row in gap_df.head(20).iterrows():
            gap_data.append({
                "skill_code": row["力量コード"],
                "skill_name": row["力量名"],
                "current_rate": round(row["現在保有率"], 3),
                "target_rate": round(row["目標保有率"], 3),
                "gap_rate": round(row["保有率ギャップ"], 3),
                "gap_percentage": round(row["保有率ギャップ率"], 3)
            })

        # Format critical skills
        critical_skills_data = []
        for idx, row in critical_skills.head(10).iterrows():
            critical_skills_data.append({
                "skill_code": row["力量コード"],
                "skill_name": row["力量名"],
                "current_rate": round(row["現在保有率"], 3),
                "target_rate": round(row["目標保有率"], 3),
                "gap_rate": round(row["保有率ギャップ"], 3),
                "gap_percentage": round(row["保有率ギャップ率"], 3)
            })

        return {
            "success": True,
            "percentile": request.percentile,
            "gap_analysis": gap_data,
            "critical_skills": critical_skills_data,
            "summary": {
                "total_skills_analyzed": len(gap_df),
                "critical_skills_count": len(critical_skills),
                "avg_gap_rate": round(gap_df["保有率ギャップ率"].mean(), 3)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Skill gap analysis failed: {str(e)}")


@router.post("/organizational/succession")
async def find_succession_candidates(request: SuccessionPlanningRequest):
    """
    Find succession candidates for a target position.
    """
    try:
        transformed_data = load_session_data(request.session_id)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]
        members_df = transformed_data["members_clean"]

        # Initialize succession planner
        planner = SuccessionPlanner()

        # Calculate position skill profile
        profile = planner.calculate_position_skill_profile(
            request.target_position,
            members_df,
            member_competence_df,
            competence_master_df,
            position_column="役職"
        )

        # Find candidates
        candidates = planner.find_succession_candidates(
            request.target_position,
            members_df,
            member_competence_df,
            competence_master_df,
            position_column="役職",
            grade_column="職能・等級",
            exclude_current_holders=True,
            max_candidates=request.max_candidates
        )

        # Format candidates for frontend
        candidates_data = []
        for idx, row in candidates.head(request.max_candidates).iterrows():
            timeline = planner.estimate_development_timeline(row['不足スキル数'])

            candidates_data.append({
                "member_code": row["メンバーコード"],
                "member_name": row["メンバー名"],
                "current_position": row.get("現在の役職", ""),
                "current_grade": row.get("現在の等級", ""),
                "readiness_score": round(row["準備度スコア"], 3),
                "skill_match_rate": round(row["スキルマッチ度"], 3),
                "owned_skills_count": int(row["保有スキル数"]),
                "missing_skills_count": int(row["不足スキル数"]),
                "estimated_timeline": timeline
            })

        return {
            "success": True,
            "target_position": request.target_position,
            "candidates": candidates_data,
            "profile": {
                "required_skills_count": len(profile) if profile is not None else 0
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Succession planning failed: {str(e)}")


@router.post("/organizational/simulate")
async def simulate_organization_changes(request: OrganizationSimulationRequest):
    """
    Simulate organizational changes (member transfers).
    """
    try:
        transformed_data = load_session_data(request.session_id)

        member_competence_df = transformed_data["member_competence"]
        competence_master_df = transformed_data["competence_master"]
        members_df = transformed_data["members_clean"]

        # Initialize simulator
        simulator = OrganizationSimulator()

        # Capture current state
        current_state = simulator.capture_current_state(
            members_df,
            member_competence_df,
            competence_master_df,
            group_by=request.group_column
        )

        # Apply transfers
        for transfer in request.transfers:
            simulator.simulate_transfer(
                transfer["member_code"],
                transfer["from_group"],
                transfer["to_group"],
                group_column=request.group_column
            )

        # Execute simulation
        simulated_state = simulator.execute_simulation(competence_master_df)

        # Compare states
        comparison_df = simulator.compare_states()

        # Calculate balance scores
        current_balance = simulator.calculate_balance_score(current_state)
        simulated_balance = simulator.calculate_balance_score(simulated_state)

        # Format comparison data
        comparison_data = []
        for idx, row in comparison_df.iterrows():
            comparison_data.append({
                "group": row.get("グループ", ""),
                "current_members": int(row.get("現在のメンバー数", 0)),
                "simulated_members": int(row.get("シミュレーション後のメンバー数", 0)),
                "current_avg_skills": round(row.get("現在の平均スキル数", 0), 1),
                "simulated_avg_skills": round(row.get("シミュレーション後の平均スキル数", 0), 1),
                "member_change": int(row.get("メンバー数変化", 0)),
                "skill_change": round(row.get("平均スキル数変化", 0), 1)
            })

        return {
            "success": True,
            "transfers_applied": len(request.transfers),
            "comparison": comparison_data,
            "balance_scores": {
                "current": round(current_balance, 3),
                "simulated": round(simulated_balance, 3),
                "change": round(simulated_balance - current_balance, 3)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Organization simulation failed: {str(e)}")
