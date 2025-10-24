"""
機械学習ベースの推薦エンジン

Matrix Factorization + 多様性再ランキングを統合した推薦エンジン
"""

import pandas as pd
from typing import List, Optional, Dict
from skillnote_recommendation.core.models import Recommendation
from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel
from skillnote_recommendation.ml.diversity import DiversityReranker


class MLRecommender:
    """機械学習ベース推薦エンジン"""

    def __init__(
        self,
        mf_model: MatrixFactorizationModel,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        diversity_reranker: Optional[DiversityReranker] = None
    ):
        """
        初期化

        Args:
            mf_model: 学習済みMatrix Factorizationモデル
            competence_master: 力量マスタ
            member_competence: 会員習得力量データ
            diversity_reranker: 多様性再ランキング器（Noneの場合はデフォルト生成）
        """
        self.mf_model = mf_model
        self.competence_master = competence_master
        self.member_competence = member_competence
        self.diversity_reranker = diversity_reranker or DiversityReranker()

        # 会員ごとの習得力量をキャッシュ
        self._member_acquired_cache = {}

    @classmethod
    def build(
        cls,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame
    ):
        """
        学習用ヘルパー
        member_competence から行列分解モデルを学習し
        学習済みモデルを使った MLRecommender を返す
        """
        # 行列分解モデルを生成
        mf_model = MatrixFactorizationModel()

        # 学習を実行
        # fitは「メンバーコード」「力量コード」「正規化レベル」を使って学習することを想定（推測です）
        mf_model.fit(member_competence)

        # インスタンス化して返却
        return cls(
            mf_model=mf_model,
            competence_master=competence_master,
            member_competence=member_competence,
            diversity_reranker=DiversityReranker()
        )

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[str] = None,
        category_filter: Optional[str] = None,
        use_diversity: bool = True,
        diversity_strategy: str = 'hybrid'
    ) -> List[Recommendation]:
        """
        力量を推薦

        Args:
            member_code: 会員コード
            top_n: 推薦件数
            competence_type: 力量タイプフィルタ（None/SKILL/EDUCATION/LICENSE）
            category_filter: カテゴリフィルタ（部分一致）
            use_diversity: 多様性再ランキングを使用するか
            diversity_strategy: 多様性戦略 ('mmr', 'category', 'type', 'hybrid')

        Returns:
            推薦結果のリスト
        """
        # 既習得力量を取得
        acquired_competences = self._get_acquired_competences(member_code)

        # Matrix Factorizationで候補を生成（多めに取得）
        candidate_size = top_n * 3 if use_diversity else top_n
        candidates = self.mf_model.predict_top_k(
            member_code=member_code,
            k=candidate_size,
            exclude_acquired=True,
            acquired_competences=acquired_competences
        )

        # 力量情報を結合
        candidates_with_info = []
        for comp_code, score in candidates:
            comp_info = self.competence_master[
                self.competence_master['力量コード'] == comp_code
            ]
            if len(comp_info) > 0:
                candidates_with_info.append((comp_code, score, comp_info.iloc[0]))

        # フィルタリング
        filtered_candidates = []
        for comp_code, score, comp_info in candidates_with_info:
            # タイプフィルタ
            if competence_type and comp_info['力量タイプ'] != competence_type:
                continue

            # カテゴリフィルタ
            if category_filter:
                category = str(comp_info.get('力量カテゴリー名', ''))
                if category_filter.lower() not in category.lower():
                    continue

            filtered_candidates.append((comp_code, score))

        # 多様性再ランキング
        if use_diversity and len(filtered_candidates) > 0:
            if diversity_strategy == 'mmr':
                final_candidates = self.diversity_reranker.rerank_mmr(
                    filtered_candidates, self.competence_master, k=top_n
                )
            elif diversity_strategy == 'category':
                final_candidates = self.diversity_reranker.rerank_category_diversity(
                    filtered_candidates, self.competence_master, k=top_n
                )
            elif diversity_strategy == 'type':
                final_candidates = self.diversity_reranker.rerank_type_diversity(
                    filtered_candidates, self.competence_master, k=top_n
                )
            elif diversity_strategy == 'hybrid':
                final_candidates = self.diversity_reranker.rerank_hybrid(
                    filtered_candidates, self.competence_master, k=top_n
                )
            else:
                final_candidates = filtered_candidates[:top_n]
        else:
            final_candidates = filtered_candidates[:top_n]

        # Recommendationオブジェクトに変換
        recommendations = []
        for comp_code, ml_score in final_candidates:
            comp_info = self.competence_master[
                self.competence_master['力量コード'] == comp_code
            ].iloc[0]

            # MLスコアを0-10スケールに変換
            priority_score = self._normalize_score(ml_score, final_candidates)

            # 推薦理由を生成
            reason = self._generate_reason(
                comp_info,
                ml_score,
                use_diversity,
                diversity_strategy
            )

            recommendation = Recommendation(
                competence_code=comp_code,
                competence_name=comp_info['力量名'],
                competence_type=comp_info['力量タイプ'],
                category=comp_info.get('力量カテゴリー名', ''),
                priority_score=priority_score,
                category_importance=0.0,  # MLモデルでは個別スコアは未使用
                acquisition_ease=0.0,
                popularity=0.0,
                reason=reason
            )
            recommendations.append(recommendation)

        return recommendations

    def _get_acquired_competences(self, member_code: str) -> List[str]:
        """会員の習得力量コードリストを取得（キャッシュ付き）"""
        if member_code not in self._member_acquired_cache:
            acquired = self.member_competence[
                self.member_competence['メンバーコード'] == member_code
            ]['力量コード'].unique().tolist()
            self._member_acquired_cache[member_code] = acquired

        return self._member_acquired_cache[member_code]

    def _normalize_score(self, score: float, all_candidates: List[tuple]) -> float:
        """
        MLスコアを0-10スケールに正規化

        Args:
            score: 元のスコア
            all_candidates: 全候補のリスト

        Returns:
            0-10スケールのスコア
        """
        if not all_candidates:
            return 5.0

        scores = [s for _, s in all_candidates]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return 5.0

        normalized = ((score - min_score) / (max_score - min_score)) * 10.0
        return round(normalized, 2)

    def _generate_reason(
        self,
        comp_info: pd.Series,
        ml_score: float,
        use_diversity: bool,
        diversity_strategy: str
    ) -> str:
        """
        推薦理由を生成
        """
        comp_name = comp_info['力量名']
        comp_type = comp_info['力量タイプ']
        category = comp_info.get('力量カテゴリー名', '')

        # 基本の推薦理由
        if comp_type == 'SKILL':
            reason = f"あなたの習得パターンから、{comp_name}の習得が推奨されます。"
        elif comp_type == 'EDUCATION':
            reason = f"研修「{comp_name}」の受講が、あなたのキャリアに適しています。"
        else:  # LICENSE
            reason = f"資格「{comp_name}」の取得が、あなたのスキルセットを強化します。"

        # カテゴリ情報
        if category:
            reason += f" （カテゴリ: {category}）"

        # 多様性戦略の説明
        if use_diversity:
            if diversity_strategy == 'hybrid':
                reason += " バランスの取れたスキル習得を考慮して選定されました。"
            elif diversity_strategy == 'mmr':
                reason += " 多様性を重視して選定されました。"
            elif diversity_strategy == 'category':
                reason += " 様々な分野をカバーするよう選定されました。"
            elif diversity_strategy == 'type':
                reason += " スキル・研修・資格のバランスを考慮して選定されました。"

        return reason

    def calculate_diversity_metrics(
        self,
        recommendations: List[Recommendation]
    ) -> Dict[str, float]:
        """
        推薦結果の多様性指標を計算
        """
        # Recommendationオブジェクトを (力量コード, スコア) のタプルに変換
        candidates = [
            (rec.competence_code, rec.priority_score)
            for rec in recommendations
        ]

        return self.diversity_reranker.calculate_diversity_metrics(
            candidates,
            self.competence_master
        )
