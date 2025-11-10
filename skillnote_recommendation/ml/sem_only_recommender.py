"""
SEM専用推薦エンジン

NMFを使用せず、構造方程式モデリング (SEM) のみで推薦を行います。
力量（スキル、資格、教育）の構造をSEMで分析し、
メンバーの現在の習得状況から次に取るべき力量を推薦します。
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel
from skillnote_recommendation.core.models import Recommendation

logger = logging.getLogger(__name__)


@dataclass
class SEMRecommendation:
    """SEM推薦結果"""

    competence_code: str
    competence_name: str
    competence_type: str
    category: str
    domain: str
    sem_score: float  # SEMスコア（0-1）
    current_level: str  # メンバーの現在のレベル（未習得/初級/中級/上級）
    target_level: str  # 推薦される目標レベル
    path_coefficient: float  # パス係数
    is_significant: bool  # 統計的に有意か
    reason: str  # 推薦理由


class SEMOnlyRecommender:
    """
    SEM専用推薦エンジン

    構造方程式モデリング（SEM）を使用して、
    力量（スキル、資格、教育）の習得構造を分析し、
    メンバーの現在の習得状況から次に取るべき力量を推薦します。

    使用例:
        recommender = SEMOnlyRecommender(
            member_competence_df=member_competence,
            competence_master_df=competence_master,
            member_master_df=member_master
        )

        # メンバーの推薦を取得
        recommendations = recommender.recommend(
            member_code="M001",
            top_n=10
        )
    """

    def __init__(
        self,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        member_master_df: pd.DataFrame,
        num_domain_categories: int = 8,
        confidence_level: float = 0.95,
    ):
        """
        初期化

        Args:
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタデータ
            member_master_df: メンバーマスタデータ
            num_domain_categories: スキル領域の分類数（5～10推奨）
            confidence_level: 信頼区間のレベル（0.95 = 95%）
        """
        self.member_competence_df = member_competence_df.copy()
        self.competence_master_df = competence_master_df.copy()
        self.member_master_df = member_master_df.copy()

        # SEMモデルを初期化
        logger.info("Initializing SkillDomainSEMModel...")
        self.sem_model = SkillDomainSEMModel(
            member_competence_df=member_competence_df,
            competence_master_df=competence_master_df,
            num_domain_categories=num_domain_categories,
            confidence_level=confidence_level,
        )

        logger.info(
            f"SEMOnlyRecommender initialized with {len(self.sem_model.get_all_domains())} domains"
        )

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[List[str]] = None,
        domain_filter: Optional[str] = None,
        min_significance: bool = True,
    ) -> List[SEMRecommendation]:
        """
        メンバーへの力量推薦を実行

        Args:
            member_code: メンバーコード
            top_n: 上位N件を返す
            competence_type: 力量タイプフィルタ（SKILL, EDUCATION, LICENSE）
            domain_filter: 領域フィルタ
            min_significance: 統計的に有意なパス係数のみを使用

        Returns:
            SEMRecommendationオブジェクトのリスト
        """
        # メンバーの現在の習得力量を取得
        member_competences = self._get_member_competences(member_code)
        member_competence_codes = set(member_competences['力量コード'].values)

        # 全ての未習得力量を取得
        all_competences = self.competence_master_df.copy()

        # 既に習得済みの力量を除外
        unacquired_competences = all_competences[
            ~all_competences['力量コード'].isin(member_competence_codes)
        ]

        # 力量タイプでフィルタリング
        if competence_type:
            before_filter = len(unacquired_competences)
            unacquired_competences = unacquired_competences[
                unacquired_competences['力量タイプ'].isin(competence_type)
            ]
            after_filter = len(unacquired_competences)
            logger.info(f"力量タイプフィルタ: {before_filter}件 → {after_filter}件（タイプ: {competence_type}）")

        logger.info(f"未習得力量数（フィルタ後）: {len(unacquired_competences)}件")

        # 各未習得力量に対してSEMスコアを計算
        recommendations = []
        skipped_by_significance = 0
        skipped_by_domain = 0

        for _, comp_row in unacquired_competences.iterrows():
            competence_code = comp_row['力量コード']

            # SEMスコアを計算
            sem_score = self.sem_model.calculate_sem_score(
                member_code=member_code,
                skill_code=competence_code
            )

            # 領域を取得
            domain = self.sem_model._find_skill_domain(competence_code)

            # 領域フィルタ
            if domain_filter and domain != domain_filter:
                skipped_by_domain += 1
                continue

            # メンバーの現在のレベルを取得
            current_level = self._get_level_name(
                self.sem_model._estimate_current_level(member_code, domain or "その他")
            )

            # 推薦理由を生成
            reason = self._generate_recommendation_reason(
                member_code=member_code,
                competence_code=competence_code,
                domain=domain,
                sem_score=sem_score,
                current_level=current_level,
            )

            # パス係数情報を取得
            path_info = self._get_path_info(domain, current_level)

            # 有意性フィルタ
            if min_significance and path_info and not path_info.get('is_significant', False):
                skipped_by_significance += 1
                continue

            recommendation = SEMRecommendation(
                competence_code=competence_code,
                competence_name=comp_row.get('力量名', competence_code),
                competence_type=comp_row.get('力量タイプ', ''),
                category=comp_row.get('力量カテゴリー名', ''),
                domain=domain or 'その他',
                sem_score=sem_score,
                current_level=current_level,
                target_level=self._get_next_level(current_level),
                path_coefficient=path_info.get('coefficient', 0.0) if path_info else 0.0,
                is_significant=path_info.get('is_significant', False) if path_info else False,
                reason=reason,
            )

            recommendations.append(recommendation)

        logger.info(f"推薦候補数（フィルタ後）: {len(recommendations)}件")
        if skipped_by_domain > 0:
            logger.info(f"  - 領域フィルタでスキップ: {skipped_by_domain}件")
        if skipped_by_significance > 0:
            logger.info(f"  - 有意性フィルタでスキップ: {skipped_by_significance}件（min_significance={min_significance}）")

        # SEMスコアでソート（降順）
        recommendations.sort(key=lambda x: x.sem_score, reverse=True)

        # 上位N件を返す
        final_recommendations = recommendations[:top_n]
        logger.info(f"最終推薦数: {len(final_recommendations)}件（top_n={top_n}）")

        return final_recommendations

    def get_member_profile(self, member_code: str) -> Dict[str, Any]:
        """
        メンバーの領域別プロファイルを取得

        Args:
            member_code: メンバーコード

        Returns:
            {
                'domains': {領域名: {潜在変数名: スコア}},
                'overall_scores': {領域名: 平均スコア},
                'acquired_competences': メンバーが持っている力量のリスト,
                'total_competences_count': 習得している力量の総数,
            }
        """
        # 領域別プロファイル
        domain_profile = self.sem_model.get_member_domain_profile(member_code)

        # 各領域の平均スコアを計算
        overall_scores = {}
        for domain, factor_scores in domain_profile.items():
            if factor_scores:
                overall_scores[domain] = np.mean(list(factor_scores.values()))
            else:
                overall_scores[domain] = 0.0

        # 習得している力量を取得
        member_competences = self._get_member_competences(member_code)

        # 力量マスタとマージして詳細情報を取得
        acquired_competences = pd.merge(
            member_competences,
            self.competence_master_df,
            on='力量コード',
            how='left'
        )

        return {
            'domains': domain_profile,
            'overall_scores': overall_scores,
            'acquired_competences': acquired_competences,
            'total_competences_count': len(member_competences),
        }

    def get_competence_gaps(
        self,
        member_code: str,
        domain: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        メンバーの力量ギャップを領域別に取得

        Args:
            member_code: メンバーコード
            domain: 領域名（指定した場合、その領域のみ返す）

        Returns:
            {
                '領域名': [
                    {
                        'competence_code': 力量コード,
                        'competence_name': 力量名,
                        'competence_type': 力量タイプ,
                        'is_acquired': 習得済みかどうか,
                        'level': メンバーのレベル（習得している場合）,
                    }
                ]
            }
        """
        # メンバーの習得力量
        member_competences = self._get_member_competences(member_code)
        member_competence_codes = set(member_competences['力量コード'].values)

        # 領域別にギャップを整理
        gaps_by_domain = {}

        domains = [domain] if domain else self.sem_model.get_all_domains()

        for domain_name in domains:
            domain_struct = self.sem_model.domain_structures.get(domain_name)
            if not domain_struct:
                continue

            # この領域の全スキルを取得
            domain_skills = []
            for latent_factor in domain_struct.latent_factors:
                domain_skills.extend(latent_factor.observed_skills)

            # スキルごとにギャップ情報を作成
            gap_info = []
            for skill_code in domain_skills:
                is_acquired = skill_code in member_competence_codes

                # スキル情報を取得
                skill_info = self.competence_master_df[
                    self.competence_master_df['力量コード'] == skill_code
                ]

                if len(skill_info) == 0:
                    continue

                skill_row = skill_info.iloc[0]

                # 習得レベルを取得
                level = None
                if is_acquired:
                    member_skill = member_competences[
                        member_competences['力量コード'] == skill_code
                    ]
                    if len(member_skill) > 0:
                        level = member_skill.iloc[0].get('正規化レベル', 0)

                gap_info.append({
                    'competence_code': skill_code,
                    'competence_name': skill_row.get('力量名', skill_code),
                    'competence_type': skill_row.get('力量タイプ', ''),
                    'category': skill_row.get('力量カテゴリー名', ''),
                    'is_acquired': is_acquired,
                    'level': level,
                })

            gaps_by_domain[domain_name] = gap_info

        return gaps_by_domain

    def get_all_domains(self) -> List[str]:
        """全領域名を取得"""
        return self.sem_model.get_all_domains()

    def get_domain_info(self, domain_name: str) -> Dict[str, Any]:
        """領域の詳細情報を取得"""
        return self.sem_model.get_domain_info(domain_name)

    def get_model_fit_indices(self, domain_name: str) -> Dict[str, float]:
        """
        モデル適合度指標を取得

        Returns:
            Dict containing:
            - avg_path_coefficient: 平均パス係数
            - significant_paths: 有意なパス数
            - total_paths: 総パス数
            - avg_loading: 平均因子負荷量
            - avg_effect_size: 平均効果サイズ（Cohen's d）
            - variance_explained: 説明分散（R²）
            - gfi: 適合度指標（GFI）
            - nfi: 規準適合度指標（NFI）
        """
        return self.sem_model.get_model_fit_indices(domain_name)

    def visualize_domain_network(
        self,
        domain_name: str,
        layout: str = "spring",
        show_all_edges: bool = False,
        min_coefficient: float = 0.0
    ):
        """
        領域のスキル依存関係をインタラクティブにプロット

        Args:
            domain_name: 領域名
            layout: レイアウト手法 ("spring", "circular", "hierarchical")
            show_all_edges: すべてのエッジを表示（有意でないものも含む）
            min_coefficient: 表示する最小パス係数（絶対値）
        """
        return self.sem_model.visualize_domain_network(
            domain_name=domain_name,
            layout=layout,
            show_all_edges=show_all_edges,
            min_coefficient=min_coefficient
        )

    def _get_member_competences(self, member_code: str) -> pd.DataFrame:
        """メンバーの習得力量を取得"""
        return self.member_competence_df[
            self.member_competence_df['メンバーコード'] == member_code
        ].copy()

    def _get_level_name(self, level: int) -> str:
        """レベル番号をレベル名に変換"""
        level_names = {
            -1: "未習得",
            0: "初級",
            1: "中級",
            2: "上級",
        }
        return level_names.get(level, "未習得")

    def _get_next_level(self, current_level: str) -> str:
        """次のレベルを取得"""
        level_progression = {
            "未習得": "初級",
            "初級": "中級",
            "中級": "上級",
            "上級": "エキスパート",
        }
        return level_progression.get(current_level, "初級")

    def _get_path_info(self, domain: Optional[str], current_level: str) -> Optional[Dict[str, Any]]:
        """パス係数情報を取得"""
        if not domain:
            return None

        domain_struct = self.sem_model.domain_structures.get(domain)
        if not domain_struct:
            return None

        # 現在のレベルに対応するパス係数を取得
        level_map = {"初級": 0, "中級": 1, "上級": 2}
        level_idx = level_map.get(current_level, -1)

        if level_idx < 0 or level_idx >= len(domain_struct.latent_factors) - 1:
            return None

        current_factor = domain_struct.latent_factors[level_idx]

        for path_coef in domain_struct.path_coefficients:
            if path_coef.from_factor == current_factor.factor_name:
                return {
                    'coefficient': path_coef.coefficient,
                    'is_significant': path_coef.is_significant,
                    'p_value': path_coef.p_value,
                    't_value': path_coef.t_value,
                }

        return None

    def _generate_recommendation_reason(
        self,
        member_code: str,
        competence_code: str,
        domain: Optional[str],
        sem_score: float,
        current_level: str,
    ) -> str:
        """推薦理由を生成"""
        if not domain:
            return f"SEMスコア: {sem_score:.2f}"

        # 領域情報を取得
        domain_info = self.sem_model.get_domain_info(domain)

        # メンバープロファイルを取得
        member_profile = self.sem_model.get_member_domain_profile(member_code)
        domain_scores = member_profile.get(domain, {})

        avg_score = np.mean(list(domain_scores.values())) if domain_scores else 0.0

        # 説明文を生成
        reason_parts = []

        if current_level == "未習得":
            reason_parts.append(
                f"{domain}領域の基礎を構築するために推薦します。"
            )
        elif current_level == "初級":
            reason_parts.append(
                f"{domain}領域で初級レベルを達成済みです。"
                f"次のステップとして中級レベルのこの力量を推薦します。"
            )
        elif current_level == "中級":
            reason_parts.append(
                f"{domain}領域で中級レベルを達成済みです。"
                f"上級レベルを目指してこの力量を推薦します。"
            )
        else:
            reason_parts.append(
                f"{domain}領域で上級レベルを達成済みです。"
                f"さらなる専門性を高めるためにこの力量を推薦します。"
            )

        reason_parts.append(f"\nSEMスコア: {sem_score:.2f}")
        reason_parts.append(f"領域習得度: {avg_score*100:.0f}%")

        return " ".join(reason_parts)

    def get_recommendation_reasoning(
        self,
        member_code: str,
        competence_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        推薦理由を詳細に分析

        指定された力量がなぜ推薦されたかを、
        習得済み力量からの影響経路とともに返します。

        Args:
            member_code: メンバーコード
            competence_code: 推薦された力量コード

        Returns:
            推薦理由の詳細情報（グラフデータ含む）、または None
        """
        # 力量情報を取得
        competence_info = self.competence_master_df[
            self.competence_master_df['力量コード'] == competence_code
        ]

        if competence_info.empty:
            logger.warning(f"Competence {competence_code} not found")
            return None

        competence_row = competence_info.iloc[0]
        target_domain = competence_row.get('力量ドメイン')

        if not target_domain or target_domain not in self.sem_model.domain_structures:
            logger.warning(f"Domain {target_domain} not found for competence {competence_code}")
            return None

        # メンバーの習得済み力量を取得
        acquired_competences = self._get_member_competences(member_code)

        # 影響経路を分析
        influences = []
        total_influence = 0.0

        domain_struct = self.sem_model.domain_structures[target_domain]

        # ターゲット力量がどの潜在変数に属するかを探す
        target_latent_factor = None
        target_loading = 0.0

        for latent_factor in domain_struct.latent_factors:
            if competence_code in latent_factor.observed_skills:
                target_latent_factor = latent_factor
                target_loading = latent_factor.factor_loadings.get(competence_code, 0.5)
                break

        if not target_latent_factor:
            logger.warning(f"Target competence {competence_code} not found in domain structure")
            return None

        # 習得済み力量からの影響を計算
        for _, acq_row in acquired_competences.iterrows():
            acq_code = acq_row['力量コード']
            acq_level = acq_row.get('正規化レベル', 0.5)

            # この習得済み力量がどの潜在変数に属するかを探す
            source_latent_factor = None
            source_loading = 0.0

            for latent_factor in domain_struct.latent_factors:
                if acq_code in latent_factor.observed_skills:
                    source_latent_factor = latent_factor
                    source_loading = latent_factor.factor_loadings.get(acq_code, 0.5)
                    break

            if not source_latent_factor:
                continue

            # パス係数を探す
            path_coefficient = 0.0
            is_significant = False
            p_value = 1.0

            for path_coeff in domain_struct.path_coefficients:
                if (path_coeff.from_factor == source_latent_factor.factor_name and
                    path_coeff.to_factor == target_latent_factor.factor_name):
                    path_coefficient = path_coeff.coefficient
                    is_significant = path_coeff.is_significant
                    p_value = path_coeff.p_value
                    break

            if path_coefficient != 0.0:
                # 影響度を計算: パス係数 × 習得レベル × source_loading × target_loading
                influence = path_coefficient * acq_level * source_loading * target_loading
                total_influence += abs(influence)

                acq_info = self.competence_master_df[
                    self.competence_master_df['力量コード'] == acq_code
                ]

                influences.append({
                    'source_code': acq_code,
                    'source_name': acq_info.iloc[0]['力量名'] if not acq_info.empty else acq_code,
                    'source_type': acq_info.iloc[0].get('力量タイプ', 'UNKNOWN') if not acq_info.empty else 'UNKNOWN',
                    'source_level': acq_level,
                    'source_loading': source_loading,
                    'target_loading': target_loading,
                    'path_coefficient': path_coefficient,
                    'influence': influence,
                    'is_significant': is_significant,
                    'p_value': p_value,
                    'source_latent': source_latent_factor.factor_name,
                    'target_latent': target_latent_factor.factor_name,
                })

        # 影響度の大きい順にソート
        influences.sort(key=lambda x: abs(x['influence']), reverse=True)

        return {
            'target_code': competence_code,
            'target_name': competence_row['力量名'],
            'target_type': competence_row.get('力量タイプ', 'UNKNOWN'),
            'target_domain': target_domain,
            'target_latent': target_latent_factor.factor_name,
            'target_loading': target_loading,
            'influences': influences,
            'total_influence': total_influence,
            'num_sources': len(influences),
        }

    def visualize_recommendation_reasoning(
        self,
        member_code: str,
        competence_code: str,
        top_n: int = 5
    ):
        """
        推薦理由をグラフで可視化

        Args:
            member_code: メンバーコード
            competence_code: 推薦された力量コード
            top_n: 表示する影響経路の数（上位N件）

        Returns:
            Plotly Figure、または None
        """
        try:
            import networkx as nx
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("networkx and plotly are required for visualization")
            return None

        reasoning = self.get_recommendation_reasoning(member_code, competence_code)

        if not reasoning or not reasoning['influences']:
            logger.warning(f"No reasoning data for {competence_code}")
            return None

        # 上位N件の影響のみを使用
        top_influences = reasoning['influences'][:top_n]

        # グラフを作成
        G = nx.DiGraph()

        # ノード追加
        # ターゲット力量
        G.add_node(
            reasoning['target_code'],
            label=reasoning['target_name'],
            competence_type=reasoning['target_type'],
            node_type='target'
        )

        # 影響元の力量
        for inf in top_influences:
            G.add_node(
                inf['source_code'],
                label=inf['source_name'],
                competence_type=inf['source_type'],
                node_type='source',
                level=inf['source_level']
            )

            # エッジ追加
            G.add_edge(
                inf['source_code'],
                reasoning['target_code'],
                weight=abs(inf['influence']),
                influence=inf['influence'],
                path_coefficient=inf['path_coefficient'],
                is_significant=inf['is_significant'],
                p_value=inf['p_value']
            )

        # レイアウトを計算（階層的）
        # ソースノードを左、ターゲットノードを右に配置
        pos = {}
        source_nodes = [inf['source_code'] for inf in top_influences]
        target_node = reasoning['target_code']

        # ソースノードを縦に並べる
        for i, node in enumerate(source_nodes):
            pos[node] = (0, i)

        # ターゲットノードを右側の中央に配置
        pos[target_node] = (1, (len(source_nodes) - 1) / 2)

        # エッジを描画
        edge_traces = []
        for edge_data in top_influences:
            source = edge_data['source_code']
            target = reasoning['target_code']

            x0, y0 = pos[source]
            x1, y1 = pos[target]

            # 影響度に応じてエッジの太さを変更
            width = abs(edge_data['influence']) * 10 + 1

            # 有意性に応じて色を変更
            color = "#2E7D32" if edge_data['is_significant'] else "#BDBDBD"

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text",
                hovertext=f"影響度: {edge_data['influence']:.3f}<br>"
                         f"パス係数: {edge_data['path_coefficient']:.3f}<br>"
                         f"習得レベル: {edge_data['source_level']:.2f}<br>"
                         f"p値: {edge_data['p_value']:.4f}<br>"
                         f"有意: {'Yes' if edge_data['is_significant'] else 'No'}",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

        # ノードを描画
        node_x_source, node_y_source, node_text_source, node_hover_source = [], [], [], []
        node_x_target, node_y_target, node_text_target, node_hover_target = [], [], [], []

        # 力量タイプごとの色
        type_colors = {
            'SKILL': '#1f77b4',
            'EDUCATION': '#ff7f0e',
            'LICENSE': '#2ca02c',
            'UNKNOWN': '#7f7f7f',
        }

        # ソースノード
        for inf in top_influences:
            node = inf['source_code']
            x, y = pos[node]
            node_x_source.append(x)
            node_y_source.append(y)

            # ノードテキスト
            label = inf['source_name']
            display_label = label if len(label) <= 20 else label[:17] + "..."
            node_text_source.append(display_label)

            # ホバー情報
            hover_text = f"<b>{label}</b><br>"
            hover_text += f"力量タイプ: {inf['source_type']}<br>"
            hover_text += f"習得レベル: {inf['source_level']:.2f}<br>"
            hover_text += f"ファクターローディング: {inf['source_loading']:.3f}<br>"
            hover_text += f"→ 影響度: {inf['influence']:.3f}"
            node_hover_source.append(hover_text)

        # ターゲットノード
        x, y = pos[target_node]
        node_x_target.append(x)
        node_y_target.append(y)

        label = reasoning['target_name']
        display_label = label if len(label) <= 20 else label[:17] + "..."
        node_text_target.append(display_label)

        hover_text = f"<b>{label}</b><br>"
        hover_text += f"力量タイプ: {reasoning['target_type']}<br>"
        hover_text += f"【推薦対象】<br>"
        hover_text += f"総影響度: {reasoning['total_influence']:.3f}<br>"
        hover_text += f"影響元: {len(top_influences)}個の習得済み力量"
        node_hover_target.append(hover_text)

        # ソースノードのトレース（習得済み）
        source_trace = go.Scatter(
            x=node_x_source,
            y=node_y_source,
            mode="markers+text",
            text=node_text_source,
            textposition="middle left",
            hoverinfo="text",
            hovertext=node_hover_source,
            marker=dict(
                size=20,
                color='#4CAF50',  # 緑（習得済み）
                line_width=2,
                line_color="#ffffff",
                symbol='circle'
            ),
            name='習得済み力量'
        )

        # ターゲットノードのトレース（推薦）
        target_trace = go.Scatter(
            x=node_x_target,
            y=node_y_target,
            mode="markers+text",
            text=node_text_target,
            textposition="middle right",
            hoverinfo="text",
            hovertext=node_hover_target,
            marker=dict(
                size=30,
                color='#FF5722',  # オレンジ（推薦）
                line_width=3,
                line_color="#ffffff",
                symbol='star'
            ),
            name='推薦力量'
        )

        # Figureを作成
        fig_data = edge_traces + [source_trace, target_trace]
        fig = go.Figure(data=fig_data)

        fig.update_layout(
            title=f"📊 推薦理由: {reasoning['target_name']}",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=120, r=120, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=max(400, len(top_influences) * 80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # ヘルプテキスト
        fig.add_annotation(
            text="矢印の太さ = 影響度の大きさ、緑の矢印 = 統計的に有意",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor='center'
        )

        return fig
