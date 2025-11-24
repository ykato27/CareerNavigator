"""
階層的ベイジアン推薦システム（統合モジュール）

3層すべて（L1ベイジアンネットワーク、L2条件付き確率、L3行列分解）を統合し、
最終的なスキル推薦を生成します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from skillnote_recommendation.ml.base_recommender import BaseRecommender
from skillnote_recommendation.ml.category_hierarchy_extractor import (
    CategoryHierarchyExtractor,
    CategoryHierarchy
)
from skillnote_recommendation.ml.category_network_learner import CategoryNetworkLearner
from skillnote_recommendation.ml.conditional_probability_learner import ConditionalProbabilityLearner
from skillnote_recommendation.ml.category_wise_mf import CategoryWiseMatrixFactorization

logger = logging.getLogger(__name__)


class HierarchicalBayesianRecommender(BaseRecommender):
    """
    階層的ベイジアン推薦システム
    
    3層アーキテクチャを統合してスキル推薦を生成：
    - Layer 1: 大カテゴリのベイジアンネットワーク
    - Layer 2: P(中カテゴリ | 大カテゴリ)の条件付き確率
    - Layer 3: カテゴリ別行列分解
    """
    
    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        category_csv_path: str = None,
        skill_csv_path: str = None,
        category_df: pd.DataFrame = None,
        skill_df: pd.DataFrame = None,
        max_indegree: int = 3,
        n_components: int = 10
    ):
        """
        初期化

        Args:
            member_competence: メンバー力量データ
            competence_master: 力量マスタ
            category_csv_path: カテゴリマスタCSVのパス（オプション）
            skill_csv_path: スキルマスタCSVのパス（オプション）
            category_df: カテゴリマスタDataFrame（オプション）
            skill_df: スキルマスタDataFrame（オプション）
            max_indegree: ベイジアンネットワークの最大入次数
            n_components: 行列分解の潜在因子数
        """
        super().__init__(name="HierarchicalBayesianRecommender", interpretability_score=4)

        # データを保存
        self.member_competence = member_competence
        self.competence_master = competence_master

        self.category_csv_path = category_csv_path
        self.skill_csv_path = skill_csv_path
        self.category_df = category_df
        self.skill_df = skill_df
        self.max_indegree = max_indegree
        self.n_components = n_components
        
        # 各レイヤーのコンポーネント
        self.hierarchy_extractor: Optional[CategoryHierarchyExtractor] = None
        self.hierarchy: Optional[CategoryHierarchy] = None
        self.network_learner: Optional[CategoryNetworkLearner] = None
        self.prob_learner: Optional[ConditionalProbabilityLearner] = None
        self.mf_learner: Optional[CategoryWiseMatrixFactorization] = None
        
        # スコア統合の重み
        self.weight_l1 = 0.3
        self.weight_l2 = 0.3
        self.weight_l3 = 0.4
    
    def fit(self, min_members_per_skill: int = 5):
        """
        モデルを学習
        
        Args:
            min_members_per_skill: 学習に含めるスキルの最小保持人数
            
        Returns:
            self
        """
        logger.info("階層的ベイジアン推薦システムの学習を開始")
        
        # 1. カテゴリ階層を抽出
        logger.info("Phase 1: カテゴリ階層を抽出中")
        self.hierarchy_extractor = CategoryHierarchyExtractor(
            category_csv_path=self.category_csv_path,
            skill_csv_path=self.skill_csv_path,
            category_df=self.category_df,
            skill_df=self.skill_df
        )
        self.hierarchy = self.hierarchy_extractor.extract_hierarchy()
        
        # 2. ユーザー×スキルマトリックスを準備
        user_skills = self._prepare_user_skill_matrix()
        
        # 3. Layer 1: ベイジアンネットワークを学習
        logger.info("Phase 2: Layer 1 ベイジアンネットワークを学習中")
        try:
            self.network_learner = CategoryNetworkLearner(
                max_indegree=self.max_indegree
            )
            self.network_learner.fit(
                user_skills,
                self.hierarchy,
                aggregation_method='max',
                n_bins=3
            )
            logger.info("Layer 1 学習完了")
        except ImportError as e:
            logger.warning(f"Layer 1 初期化エラー（pgmpyが利用できません）: {e}")
            logger.info("Layer 1をスキップして学習を続行します")
            self.network_learner = None
        except Exception as e:
            logger.warning(f"Layer 1 学習エラー: {e}")
            self.network_learner = None
        
        # 4. Layer 2: 条件付き確率を学習
        logger.info("Phase 3: Layer 2 条件付き確率を学習中")
        self.prob_learner = ConditionalProbabilityLearner()
        self.prob_learner.learn_conditional_probs(self.hierarchy, user_skills)
        logger.info("Layer 2 学習完了")
        
        # 5. Layer 3: カテゴリ別行列分解を学習
        logger.info("Phase 4: Layer 3 カテゴリ別行列分解を学習中")
        self.mf_learner = CategoryWiseMatrixFactorization(
            n_components=self.n_components
        )
        self.mf_learner.fit_category_models(user_skills, self.hierarchy)
        logger.info("Layer 3 学習完了")
        
        logger.info("階層的ベイジアン推薦システムの学習完了")
        return self
    
    def _prepare_user_skill_matrix(self) -> pd.DataFrame:
        """
        ユーザー×スキルマトリックスを準備

        Returns:
            ユーザー×スキルのDataFrame
        """
        # member_competenceからSKILLタイプのみを抽出
        skill_data = self.member_competence[
            self.member_competence['力量タイプ'] == 'SKILL'
        ].copy()

        # レベルを数値型に変換（文字列として保存されている場合に対応）
        skill_data['レベル'] = pd.to_numeric(skill_data['レベル'], errors='coerce').fillna(0)

        # ピボットしてユーザー×スキルマトリックスを作成
        user_skill_matrix = skill_data.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='レベル',
            fill_value=0,
            aggfunc='mean'
        )

        return user_skill_matrix
    
    def recommend(
        self,
        member_code: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        メンバーへのスキル推薦
        
        Args:
            member_code: メンバーID
            top_n: 推薦件数
            
        Returns:
            推薦結果のリスト
        """
        if self.hierarchy is None:
            raise ValueError("先にfit()を実行してください")
        
        logger.info(f"メンバー {member_code} への推薦を生成中")
        
        # ユーザーの保有スキルを取得
        user_skills = self._get_user_skills(member_code)
        
        # 候補スキルを取得（保有していないスキル）
        all_skills = set(self.hierarchy.skill_to_category.keys())
        owned_skills = set(user_skills.keys())
        candidate_skills = list(all_skills - owned_skills)
        
        # 各候補スキルのスコアを計算
        recommendations = []
        for skill_code in candidate_skills:
            score, explanation = self._calculate_skill_score(
                member_code,
                skill_code,
                user_skills
            )
            
            if score > 0:
                # スキル情報を取得
                skill_info = self._get_skill_info(skill_code)
                
                recommendations.append({
                    '力量コード': skill_code,
                    '力量名': skill_info.get('力量名', skill_code),
                    'スコア': score,
                    '説明': explanation,
                    'カテゴリ': skill_info.get('カテゴリ', '')
                })
        
        # スコアでソート
        recommendations.sort(key=lambda x: x['スコア'], reverse=True)
        
        logger.info(f"{len(recommendations)}件の候補から上位{top_n}件を推薦")
        
        return recommendations[:top_n]
    
    def _get_user_skills(self, member_code: str) -> Dict[str, float]:
        """
        ユーザーの保有スキルとレベルを取得

        Args:
            member_code: メンバーコード

        Returns:
            スキルコードとレベルの辞書
        """
        user_data = self.member_competence[
            (self.member_competence['メンバーコード'] == member_code) &
            (self.member_competence['力量タイプ'] == 'SKILL')
        ].copy()

        # レベルを数値型に変換（文字列として保存されている場合に対応）
        user_data['レベル'] = pd.to_numeric(user_data['レベル'], errors='coerce').fillna(0)

        return dict(zip(user_data['力量コード'], user_data['レベル']))
    
    def _calculate_skill_score(
        self,
        member_code: str,
        skill_code: str,
        user_skills: Dict[str, float]
    ) -> tuple[float, str]:
        """
        スキルの推薦スコアを計算

        Args:
            member_code: メンバーコード
            skill_code: スキルコード
            user_skills: ユーザーの保有スキル

        Returns:
            (スコア, 説明文)のタプル
        """
        # スキルのカテゴリを取得
        if skill_code not in self.hierarchy.skill_to_category:
            return 0.0, ""

        category_code = self.hierarchy.skill_to_category[skill_code]

        # L1, L2カテゴリを取得
        l1_code = self.hierarchy.get_l1_category(category_code)
        l2_code = self.hierarchy.get_l2_category(category_code)

        # Layer 2/3が使えない場合のフォールバック処理
        if not l1_code or not l2_code or len(self.prob_learner.conditional_probs) == 0:
            return self._calculate_fallback_score(
                member_code,
                skill_code,
                category_code,
                user_skills
            )

        # Layer 1: L1カテゴリの準備度
        l1_readiness = self._get_l1_readiness(l1_code, user_skills)

        # Layer 2: P(L2 | L1)
        l2_prob = self.prob_learner.get_conditional_prob(l1_code, l2_code)

        # Layer 3: スキルのMFスコア
        l3_score = self._get_l3_score(member_code, l2_code, skill_code)

        # スコアを統合（乗算的アプローチ）
        final_score = (
            (l1_readiness ** self.weight_l1) *
            (l2_prob ** self.weight_l2) *
            (l3_score ** self.weight_l3)
        )

        # 説明文を生成
        explanation = self._generate_explanation(
            l1_code,
            l2_code,
            l1_readiness,
            l2_prob,
            l3_score
        )

        return final_score, explanation

    def _calculate_fallback_score(
        self,
        member_code: str,
        skill_code: str,
        category_code: str,
        user_skills: Dict[str, float]
    ) -> tuple[float, str]:
        """
        Layer 2/3が使えない場合のフォールバック推薦スコア計算

        ユーザーの保有スキルに基づいて推薦スコアを計算します：
        1. 同じカテゴリに既に高レベルのスキルを持っている → 高スコア
        2. 関連カテゴリに高レベルのスキルを持っている → 中スコア
        3. 似たスキルセットを持つ他のユーザーが保有している → 中スコア

        Args:
            member_code: メンバーコード
            skill_code: スキルコード
            category_code: カテゴリコード
            user_skills: ユーザーの保有スキル

        Returns:
            (スコア, 説明文)のタプル
        """
        score = 0.0
        reasons = []

        # 1. 同じL3カテゴリ内のスキルレベルを確認
        if category_code in self.hierarchy.category_to_skills:
            same_cat_skills = [
                s for s in self.hierarchy.category_to_skills[category_code]
                if s in user_skills
            ]
            if same_cat_skills:
                avg_level = np.mean([user_skills[s] for s in same_cat_skills])
                category_score = min(avg_level / 5.0, 1.0)
                score += category_score * 0.5
                cat_name = self.hierarchy.category_names.get(category_code, category_code)
                reasons.append(f"{cat_name}で平均レベル{avg_level:.1f}")

        # 2. 親カテゴリ（L2）内のスキルレベルを確認
        parent_code = self.hierarchy.parent_mapping.get(category_code)
        if parent_code and parent_code in self.hierarchy.children_mapping:
            parent_cat_skills = []
            for child_cat in self.hierarchy.children_mapping[parent_code]:
                if child_cat in self.hierarchy.category_to_skills:
                    parent_cat_skills.extend([
                        s for s in self.hierarchy.category_to_skills[child_cat]
                        if s in user_skills
                    ])
            if parent_cat_skills:
                avg_level = np.mean([user_skills[s] for s in parent_cat_skills])
                parent_score = min(avg_level / 5.0, 1.0) * 0.8
                score += parent_score * 0.3
                parent_name = self.hierarchy.category_names.get(parent_code, parent_code)
                reasons.append(f"{parent_name}で平均レベル{avg_level:.1f}")

        # 3. 基本スコア（全体の平均レベル）
        if user_skills:
            avg_all = np.mean(list(user_skills.values()))
            base_score = min(avg_all / 5.0, 1.0) * 0.5
            score += base_score * 0.2

        # スコアを0-1の範囲に正規化
        score = min(max(score, 0.0), 1.0)

        # 説明文を生成
        cat_name = self.hierarchy.category_names.get(category_code, category_code)
        if reasons:
            explanation = f"{cat_name}に関連 - " + "、".join(reasons)
        else:
            explanation = f"{cat_name}のスキル"

        return score, explanation
    
    def _get_l1_readiness(
        self,
        l1_code: str,
        user_skills: Dict[str, float]
    ) -> float:
        """
        L1カテゴリの準備度を取得
        
        Args:
            l1_code: L1カテゴリコード
            user_skills: ユーザーの保有スキル
            
        Returns:
            準備度スコア（0.0-1.0）
        """
        if self.network_learner is None or self.network_learner.model is None:
            # ベイジアンネットワークが利用できない場合は簡易計算
            return self._simple_l1_readiness(l1_code, user_skills)
        
        # ユーザーのカテゴリレベルを計算
        user_categories = {}
        for skill_code, level in user_skills.items():
            if skill_code in self.hierarchy.skill_to_category:
                cat = self.hierarchy.skill_to_category[skill_code]
                cat_l1 = self.hierarchy.get_l1_category(cat)
                if cat_l1:
                    user_categories[cat_l1] = max(
                        user_categories.get(cat_l1, 0),
                        level
                    )
        
        # 離散化（0-2のレベルに）
        user_categories_discrete = {
            k: min(int(v / 2), 2) for k, v in user_categories.items()
        }
        
        try:
            readiness = self.network_learner.predict_category_readiness(
                user_categories_discrete,
                l1_code
            )
            return readiness
        except Exception as e:
            logger.warning(f"L1準備度計算エラー: {e}")
            return self._simple_l1_readiness(l1_code, user_skills)
    
    def _simple_l1_readiness(
        self,
        l1_code: str,
        user_skills: Dict[str, float]
    ) -> float:
        """
        簡易的なL1準備度計算
        
        Args:
            l1_code: L1カテゴリコード
            user_skills: ユーザーの保有スキル
            
        Returns:
            準備度スコア（0.0-1.0）
        """
        # このL1カテゴリのスキルの平均レベルを計算
        l1_skills = []
        for skill_code, level in user_skills.items():
            if skill_code in self.hierarchy.skill_to_category:
                cat = self.hierarchy.skill_to_category[skill_code]
                if self.hierarchy.get_l1_category(cat) == l1_code:
                    l1_skills.append(level)
        
        if not l1_skills:
            return 0.5  # デフォルト値
        
        avg_level = np.mean(l1_skills)
        return min(avg_level / 5.0, 1.0)  # 0-5 → 0-1に正規化
    
    def _get_l3_score(
        self,
        member_code: str,
        l2_code: str,
        skill_code: str
    ) -> float:
        """
        L3（MF）スコアを取得
        
        Args:
            member_code: メンバーコード
            l2_code: L2カテゴリコード
            skill_code: スキルコード
            
        Returns:
            MFスコア（0.0-1.0）
        """
        if self.mf_learner is None:
            return 0.5  # デフォルト値
        
        try:
            user_skill_matrix = self._prepare_user_skill_matrix()
            scores = self.mf_learner.predict_skill_scores(
                member_code,
                l2_code,
                user_skill_matrix
            )
            
            if skill_code in scores.index:
                score = scores[skill_code]
                # 0-5のスコアを0-1に正規化
                return min(score / 5.0, 1.0)
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"L3スコア計算エラー: {e}")
            return 0.5
    
    def _generate_explanation(
        self,
        l1_code: str,
        l2_code: str,
        l1_readiness: float,
        l2_prob: float,
        l3_score: float
    ) -> str:
        """
        階層的説明文を生成
        
        Args:
            l1_code: L1カテゴリコード
            l2_code: L2カテゴリコード
            l1_readiness: L1準備度
            l2_prob: L2条件付き確率
            l3_score: L3スコア
            
        Returns:
            説明文
        """
        l1_name = self.hierarchy.category_names.get(l1_code, l1_code)
        l2_name = self.hierarchy.category_names.get(l2_code, l2_code)
        
        explanation = (
            f"{l1_name}の準備度{l1_readiness*100:.0f}%、"
            f"{l2_name}への適合度{l2_prob*100:.0f}%、"
            f"スキル推薦度{l3_score*100:.0f}%"
        )
        
        return explanation
    
    def explain(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """
        推薦結果の説明を生成

        Args:
            recommendations: 推薦結果のリスト

        Returns:
            説明文のリスト
        """
        return [rec.get('説明', '') for rec in recommendations]

    def generate_hierarchy_graph(
        self,
        skill_code: str,
        member_code: str,
        output_path: str = "hierarchy_graph.html",
        height: str = "600px"
    ) -> str:
        """
        推薦スキルを中心とした階層グラフを生成

        Args:
            skill_code: 中心となるスキルコード
            member_code: メンバーコード
            output_path: 出力HTMLファイルパス
            height: グラフの高さ

        Returns:
            出力ファイルパス
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("pyvisがインストールされていません")
            raise ImportError("pyvis が必要です。pip install pyvis でインストールしてください。")

        # ユーザーの保有スキルを取得
        user_skills = self._get_user_skills(member_code)

        # スキル情報を取得
        skill_info = self._get_skill_info(skill_code)
        skill_name = skill_info['力量名']

        # カテゴリ情報を取得
        if skill_code not in self.hierarchy.skill_to_category:
            logger.warning(f"スキル {skill_code} のカテゴリ情報が見つかりません")
            return None

        category_code = self.hierarchy.skill_to_category[skill_code]
        l1_code = self.hierarchy.get_l1_category(category_code)
        l2_code = self.hierarchy.get_l2_category(category_code)

        # ネットワークを初期化
        net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black")
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "hierarchicalRepulsion": {
                    "centralGravity": 0.3,
                    "springLength": 150,
                    "springConstant": 0.05,
                    "nodeDistance": 200
                },
                "solver": "hierarchicalRepulsion"
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": 150
                }
            }
        }
        """)

        # L1カテゴリノードを追加
        if l1_code:
            l1_name = self.hierarchy.category_names.get(l1_code, l1_code)
            net.add_node(
                l1_code,
                label=l1_name,
                color="#e74c3c",
                size=30,
                title=f"L1カテゴリ: {l1_name}",
                level=0
            )

        # L2カテゴリノードを追加
        if l2_code:
            l2_name = self.hierarchy.category_names.get(l2_code, l2_code)
            net.add_node(
                l2_code,
                label=l2_name,
                color="#e67e22",
                size=25,
                title=f"L2カテゴリ: {l2_name}",
                level=1
            )

            # L1 -> L2 エッジ
            if l1_code:
                net.add_edge(l1_code, l2_code, color="#95a5a6")

        # L3カテゴリノードを追加
        category_name = self.hierarchy.category_names.get(category_code, category_code)
        net.add_node(
            category_code,
            label=category_name,
            color="#f39c12",
            size=20,
            title=f"L3カテゴリ: {category_name}",
            level=2
        )

        # L2 -> L3 エッジ
        if l2_code:
            net.add_edge(l2_code, category_code, color="#95a5a6")

        # 推薦スキルノードを追加（中心）
        net.add_node(
            skill_code,
            label=skill_name,
            color="#3498db",
            size=35,
            title=f"推薦スキル: {skill_name}",
            level=3
        )
        net.add_edge(category_code, skill_code, color="#3498db", width=3)

        # 同じL3カテゴリ内の他のスキルを追加
        if category_code in self.hierarchy.category_to_skills:
            sibling_skills = self.hierarchy.category_to_skills[category_code]
            # 保有スキルを優先的に表示
            owned_siblings = [s for s in sibling_skills if s in user_skills and s != skill_code]
            other_siblings = [s for s in sibling_skills if s not in user_skills and s != skill_code]

            # 保有スキルは全て表示、その他は最大10個まで
            siblings_to_show = owned_siblings + other_siblings[:10]

            for sibling_code in siblings_to_show:
                sibling_info = self._get_skill_info(sibling_code)
                sibling_name = sibling_info['力量名']

                # ユーザーが保有しているか確認
                is_owned = sibling_code in user_skills

                net.add_node(
                    sibling_code,
                    label=sibling_name,
                    color="#2ecc71" if is_owned else "#bdc3c7",
                    size=20 if is_owned else 15,
                    title=f"{'保有スキル' if is_owned else '関連スキル'}: {sibling_name}",
                    level=3
                )
                net.add_edge(
                    category_code,
                    sibling_code,
                    color="#2ecc71" if is_owned else "#95a5a6"
                )

        # L2カテゴリ配下の他のL3カテゴリとそのスキルも表示
        if l2_code and l2_code in self.hierarchy.children_mapping:
            other_l3_categories = [
                cat for cat in self.hierarchy.children_mapping[l2_code]
                if cat != category_code and cat in self.hierarchy.level3_categories
            ]

            # 最大2つの他のL3カテゴリを表示
            for other_l3_code in other_l3_categories[:2]:
                other_l3_name = self.hierarchy.category_names.get(other_l3_code, other_l3_code)
                net.add_node(
                    other_l3_code,
                    label=other_l3_name,
                    color="#f39c12",
                    size=15,
                    title=f"関連L3カテゴリ: {other_l3_name}",
                    level=2
                )
                net.add_edge(l2_code, other_l3_code, color="#95a5a6", dashes=True)

                # 各L3カテゴリから保有スキルを優先的に表示
                if other_l3_code in self.hierarchy.category_to_skills:
                    other_skills = self.hierarchy.category_to_skills[other_l3_code]
                    owned_other_skills = [s for s in other_skills if s in user_skills]
                    other_other_skills = [s for s in other_skills if s not in user_skills]

                    # 保有スキルは全て、その他は最大3個まで
                    skills_to_show = owned_other_skills + other_other_skills[:3]

                    for other_skill_code in skills_to_show:
                        other_skill_info = self._get_skill_info(other_skill_code)
                        other_skill_name = other_skill_info['力量名']
                        is_owned = other_skill_code in user_skills

                        net.add_node(
                            other_skill_code,
                            label=other_skill_name,
                            color="#2ecc71" if is_owned else "#ecf0f1",
                            size=18 if is_owned else 12,
                            title=f"{'保有スキル' if is_owned else '関連スキル'}: {other_skill_name}",
                            level=3
                        )
                        net.add_edge(
                            other_l3_code,
                            other_skill_code,
                            color="#2ecc71" if is_owned else "#bdc3c7"
                        )

        # HTMLファイルとして保存
        net.save_graph(output_path)

        return output_path

    def _get_skill_info(self, skill_code: str) -> Dict[str, str]:
        """
        スキル情報を取得
        
        Args:
            skill_code: スキルコード
            
        Returns:
            スキル情報の辞書
        """
        skill_info = self.competence_master[
            self.competence_master['力量コード'] == skill_code
        ]
        
        if len(skill_info) > 0:
            row = skill_info.iloc[0]
            category_code = self.hierarchy.skill_to_category.get(skill_code, '')
            category_name = self.hierarchy.category_names.get(category_code, '')
            
            return {
                '力量名': row.get('力量名', skill_code),
                'カテゴリ': category_name
            }
        else:
            return {
                '力量名': skill_code,
                'カテゴリ': ''
            }
