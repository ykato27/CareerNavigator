"""
Causal Recommendation統合型キャリアパス生成

因果グラフを活用して以下を実現:
1. Causalスコアによる推薦フィルタリング
2. スキル依存関係の抽出
3. 依存関係ベースのガントチャート生成
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
from skillnote_recommendation.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CompetenceGapWithCausal:
    """Causalスコア付きスキルギャップ"""
    competence_code: str
    competence_name: str
    category: str
    competence_type: str = ""
    
    # Causalスコア（正規化済み 0.0-1.0）
    readiness_score: float = 0.0      # 準備完了度
    bayesian_score: float = 0.0       # 統計的確率
    utility_score: float = 0.0        # 有用性
    total_score: float = 0.0          # 総合スコア
    
    # 推薦理由
    readiness_reasons: List[Tuple[str, float]] = field(default_factory=list)
    utility_reasons: List[Tuple[str, float]] = field(default_factory=list)
    
    # 依存関係（後で追加）
    prerequisites: List[str] = field(default_factory=list)
    enables: List[str] = field(default_factory=list)


class CausalFilteredLearningPath:
    """Causal Recommendationベースの学習パスフィルタリング"""
    
    def __init__(
        self,
        causal_recommender: CausalGraphRecommender,
        min_total_score: float = 0.2,      # 総合スコア閾値
        min_readiness_score: float = 0.05,  # Readiness最低値
    ):
        """
        Args:
            causal_recommender: 学習済みCausalGraphRecommender
            min_total_score: 推薦する最小総合スコア
            min_readiness_score: 推薦する最小準備完了度
        """
        self.causal_recommender = causal_recommender
        self.min_total_score = min_total_score
        self.min_readiness_score = min_readiness_score
    
    def generate_filtered_path(
        self,
        gap_analysis: Dict,
        member_code: str
    ) -> List[CompetenceGapWithCausal]:
        """
        ギャップ分析結果をCausalスコアでフィルタリング
        
        Args:
            gap_analysis: CareerGapAnalyzerの結果
            member_code: 対象メンバーコード
            
        Returns:
            Causalスコアが閾値以上のスキルのみ
        """
        if not self.causal_recommender.is_fitted:
            logger.warning("Causal Recommenderが学習されていません")
            return []
        
        missing_competences = gap_analysis.get("missing_competences", [])
        
        logger.info(f"ギャップ分析で抽出されたスキル数: {len(missing_competences)}")
        
        # 各スキルにCausalスコアを付与
        scored_competences = []
        all_scores = []  # デバッグ用：全スコアを記録
        
        for comp_info in missing_competences:
            # CausalGraphRecommenderから3軸スコアを取得
            causal_score = self.causal_recommender.get_score_for_skill(
                member_code=member_code,
                skill_code=comp_info["competence_code"]
            )
            
            all_scores.append(causal_score["total_score"])  # デバッグ用
            
            # フィルタリング条件
            if (causal_score["total_score"] >= self.min_total_score and
                causal_score["readiness"] >= self.min_readiness_score):
                
                scored_competences.append(
                    CompetenceGapWithCausal(
                        competence_code=comp_info["competence_code"],
                        competence_name=comp_info["competence_name"],
                        category=comp_info.get("category", "その他"),
                        competence_type=comp_info.get("competence_type", ""),
                        # Causalスコア
                        readiness_score=causal_score["readiness"],
                        bayesian_score=causal_score["bayesian"],
                        utility_score=causal_score["utility"],
                        total_score=causal_score["total_score"],
                        # 推薦理由
                        readiness_reasons=causal_score["readiness_reasons"],
                        utility_reasons=causal_score["utility_reasons"],
                    )
                )
        
        # 総合スコアでソート
        scored_competences.sort(key=lambda x: x.total_score, reverse=True)
        
        # デバッグ情報をログ出力
        if all_scores:
            logger.info(f"Causalスコア範囲: {min(all_scores):.3f} - {max(all_scores):.3f}")
            logger.info(f"平均スコア: {sum(all_scores)/len(all_scores):.3f}")
        logger.info(f"Causalフィルタリング後: {len(scored_competences)}スキル")
        
        # スコアが全て0または極端に低い場合の対応
        if len(scored_competences) == 0 and len(missing_competences) > 0:
            logger.warning("フィルタリング後のスキルが0件です。閾値を無視して全スキルを返します。")
            # 全スキルにCausalスコアを付与して返す（閾値無視）
            for comp_info in missing_competences:
                causal_score = self.causal_recommender.get_score_for_skill(
                    member_code=member_code,
                    skill_code=comp_info["competence_code"]
                )
                
                scored_competences.append(
                    CompetenceGapWithCausal(
                        competence_code=comp_info["competence_code"],
                        competence_name=comp_info["competence_name"],
                        category=comp_info.get("category", "その他"),
                        competence_type=comp_info.get("competence_type", ""),
                        readiness_score=causal_score["readiness"],
                        bayesian_score=causal_score["bayesian"],
                        utility_score=causal_score["utility"],
                        total_score=causal_score["total_score"],
                        readiness_reasons=causal_score["readiness_reasons"],
                        utility_reasons=causal_score["utility_reasons"],
                    )
                )
            scored_competences.sort(key=lambda x: x.total_score, reverse=True)
        
        return scored_competences


class DependencyAnalyzer:
    """因果グラフに基づくスキル依存関係分析"""
    
    def __init__(
        self,
        causal_recommender: CausalGraphRecommender,
        min_effect_threshold: float = 0.05,  # 依存と見なす因果効果の閾値
    ):
        """
        Args:
            causal_recommender: 学習済みCausalGraphRecommender
            min_effect_threshold: 依存関係と見なす最小因果効果
        """
        self.causal_recommender = causal_recommender
        self.min_effect_threshold = min_effect_threshold
    
    def extract_dependencies(
        self,
        competences: List[CompetenceGapWithCausal],
        competence_master: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        スキル間の依存関係を抽出
        
        Args:
            competences: Causalフィルタリング済みスキルリスト
            competence_master: 力量マスタ（スキル名取得用）
            
        Returns:
            {
                skill_code: {
                    "prerequisites": [(skill_code, effect), ...],  # このスキルの前提
                    "enables": [(skill_code, effect), ...],        # このスキルが開く道
                }
            }
        """
        if not self.causal_recommender.is_fitted:
            logger.warning("Causal Recommenderが学習されていません")
            return {}
        
        dependencies = {}
        skill_codes = {c.competence_code: c.competence_name for c in competences}
        
        # コードから名前へのマッピング
        code_to_name = dict(zip(
            competence_master['力量コード'],
            competence_master['力量名']
        ))
        
        for comp in competences:
            prerequisites = []  # このスキルを学ぶ前提
            enables = []        # このスキルが役立つ先
            
            comp_name = code_to_name.get(comp.competence_code, comp.competence_name)
            
            # 他のスキルとの因果関係をチェック
            for other_comp in competences:
                if comp.competence_code == other_comp.competence_code:
                    continue
                
                other_name = code_to_name.get(other_comp.competence_code, other_comp.competence_name)
                
                # other → comp の因果効果（他スキルがこのスキルの前提）
                effect_to_this = self.causal_recommender.get_effect(other_name, comp_name)
                if effect_to_this >= self.min_effect_threshold:
                    prerequisites.append((other_comp.competence_code, effect_to_this))
                
                # comp → other の因果効果（このスキルが他スキルに役立つ）
                effect_from_this = self.causal_recommender.get_effect(comp_name, other_name)
                if effect_from_this >= self.min_effect_threshold:
                    enables.append((other_comp.competence_code, effect_from_this))
            
            # 因果効果の強い順にソート
            prerequisites.sort(key=lambda x: x[1], reverse=True)
            enables.sort(key=lambda x: x[1], reverse=True)
            
            dependencies[comp.competence_code] = {
                "prerequisites": prerequisites,
                "enables": enables,
            }
        
        logger.info(f"{len(dependencies)}スキルの依存関係を抽出しました")
        
        return dependencies


class SmartRoadmapVisualizer:
    """依存関係を考慮したロードマップ可視化"""
    
    def create_dependency_based_roadmap(
        self,
        competences: List[CompetenceGapWithCausal],
        dependencies: Dict[str, Dict],
        target_member_name: str = "未設定"
    ) -> go.Figure:
        """
        依存関係ベースのガントチャートを作成
        
        Args:
            competences: Causalフィルタリング済みスキルリスト
            dependencies: 依存関係情報
            target_member_name: 目標メンバー名
            
        Returns:
            Plotlyガントチャート
        """
        if not competences:
            logger.warning("スキルリストが空です")
            return go.Figure()
        
        # トポロジカルソートで学習順序を決定
        learning_order = self._topological_sort(competences, dependencies)
        
        # 各スキルの開始時間と期間を計算
        schedule = self._calculate_schedule(learning_order, competences, dependencies)
        
        # Plotlyガントチャート用のデータ構造を作成
        tasks = []
        colors = []
        
        for skill_code, timing in schedule.items():
            comp = next((c for c in competences if c.competence_code == skill_code), None)
            if not comp:
                continue
            
            task_dict = dict(
                Task=comp.competence_name,
                Start=timing["start_week"],  # 週数（整数）
                Finish=timing["end_week"],   # 週数（整数）
                Duration=timing["duration_weeks"],  # 期間（週）
                Resource=f"スコア: {comp.total_score:.2f}"
            )
            tasks.append(task_dict)
            
            # スコアで色分け
            if comp.total_score >= 0.7:
                colors.append('#2ecc71')  # 高優先度: 緑
            elif comp.total_score >= 0.4:
                colors.append('#3498db')  # 中優先度: 青
            else:
                colors.append('#95a5a6')  # 低優先度: グレー
        
        # Plotlyでガントチャートを生成
        fig = go.Figure()
        
        for i, task in enumerate(tasks):
            fig.add_trace(go.Bar(
                x=[task['Duration']],  # 期間（週）
                y=[task['Task']],
                base=[task['Start']],  # 開始週
                orientation='h',
                marker=dict(color=colors[i]),
                name=task['Resource'],
                text=task['Resource'],
                textposition='inside',
                hovertemplate=(
                    f"<b>{task['Task']}</b><br>"
                    f"{task['Resource']}<br>"
                    f"開始: Week {task['Start']}<br>"
                    f"完了予定: Week {task['Finish']}<br>"
                    f"期間: {task['Duration']}週間<br>"
                    "<extra></extra>"
                ),
            ))
        
        fig.update_layout(
            title=f"📅 キャリアロードマップ - {target_member_name}",
            xaxis_title="週",
            yaxis_title="スキル",
            height=max(400, len(tasks) * 40),
            showlegend=False,
            barmode='overlay',
            plot_bgcolor='rgba(240,240,240,0.5)',
        )
        
        return fig
    
    def _topological_sort(
        self,
        competences: List[CompetenceGapWithCausal],
        dependencies: Dict[str, Dict]
    ) -> List[List[str]]:
        """
        トポロジカルソートで学習順序を決定
        
        Returns:
            [[layer1_skills], [layer2_skills], ...] 
            各レイヤー内のスキルは並列学習可能
        """
        # グラフ構造を構築
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        all_skills = {c.competence_code for c in competences}
        
        for comp in competences:
            skill_code = comp.competence_code
            prereqs = dependencies.get(skill_code, {}).get("prerequisites", [])
            in_degree[skill_code] = len(prereqs)
            
            enables = dependencies.get(skill_code, {}).get("enables", [])
            for next_skill, _ in enables:
                if next_skill in all_skills:
                    graph[skill_code].append(next_skill)
        
        # Kahnのアルゴリズムでトポロジカルソート
        layers = []
        current_layer = [s for s in all_skills if in_degree[s] == 0]
        
        while current_layer:
            layers.append(sorted(current_layer))  # 安定ソートのため
            next_layer = []
            
            for skill in current_layer:
                for next_skill in graph[skill]:
                    in_degree[next_skill] -= 1
                    if in_degree[next_skill] == 0:
                        next_layer.append(next_skill)
            
            current_layer = next_layer
        
        logger.info(f"トポロジカルソート完了: {len(layers)}レイヤー")
        
        return layers
    
    def _calculate_schedule(
        self,
        learning_order: List[List[str]],
        competences: List[CompetenceGapWithCausal],
        dependencies: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        各スキルの開始・終了時間を計算
        
        Returns:
            {
                skill_code: {
                    "start_week": int,
                    "end_week": int,
                    "duration_weeks": int,
                    "start_date": datetime,
                    "end_date": datetime
                }
            }
        """
        schedule = {}
        current_week = 0
        base_date = datetime.now()
        
        for layer_idx, layer in enumerate(learning_order):
            # このレイヤーの最大期間を計算
            max_duration = 0
            for skill_code in layer:
                duration = self._estimate_duration(skill_code, dependencies)
                max_duration = max(max_duration, duration)
            
            # レイヤー内の全スキルに同じ開始時間を設定（並列学習）
            for skill_code in layer:
                duration = self._estimate_duration(skill_code, dependencies)
                start_date = base_date + timedelta(weeks=current_week)
                end_date = start_date + timedelta(weeks=duration)
                
                schedule[skill_code] = {
                    "start_week": current_week,
                    "end_week": current_week + duration,
                    "duration_weeks": duration,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            
            # 次のレイヤーは現在のレイヤーの最長スキル完了後に開始
            current_week += max_duration
        
        return schedule
    
    def _estimate_duration(
        self,
        skill_code: str,
        dependencies: Dict[str, Dict]
    ) -> int:
        """
        スキル習得にかかる期間を推定（週単位）
        
        基本: 2週間
        前提スキルが多い: +前提スキル数
        """
        base_duration = 2
        prereq_count = len(dependencies.get(skill_code, {}).get("prerequisites", []))
        return base_duration + min(prereq_count, 4)  # 最大6週間
