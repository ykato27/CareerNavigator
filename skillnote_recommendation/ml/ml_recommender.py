"""
機械学習ベースの推薦エンジン（NMF対応版）

DataFrameベースのMatrixFactorizationModelと整合する設計。
"""

import pandas as pd
from typing import List, Optional, Dict
from skillnote_recommendation.core.models import Recommendation
from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel
from skillnote_recommendation.ml.diversity import DiversityReranker
from skillnote_recommendation.ml.exceptions import ColdStartError, MLModelNotTrainedError


class MLRecommender:
    """機械学習ベース推薦エンジン（NMF版）"""

    def __init__(
        self,
        mf_model: MatrixFactorizationModel,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        diversity_reranker: Optional[DiversityReranker] = None
    ):
        self.mf_model = mf_model
        self.competence_master = competence_master
        self.member_competence = member_competence
        self.diversity_reranker = diversity_reranker or DiversityReranker()
        self._member_acquired_cache = {}

    # =========================================================
    # 学習
    # =========================================================
    @classmethod
    def build(
        cls,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame
    ):
        """
        member_competence（会員習得力量データ）から会員×力量マトリクスを生成し、
        MatrixFactorizationModel（NMF）を学習。
        """
        print("\n" + "=" * 80)
        print("MLモデル学習開始（NMF）")
        print("=" * 80)

        # 会員×力量マトリクスを作成
        skill_matrix = member_competence.pivot_table(
            index="メンバーコード",
            columns="力量コード",
            values="正規化レベル",
            fill_value=0
        )

        # NMFモデルを学習
        mf_model = MatrixFactorizationModel(n_components=20, random_state=42)
        mf_model.fit(skill_matrix)

        print(f"会員数: {skill_matrix.shape[0]}")
        print(f"力量数: {skill_matrix.shape[1]}")
        print(f"再構成誤差: {mf_model.get_reconstruction_error():.4f}")
        print("=" * 80)

        return cls(
            mf_model=mf_model,
            competence_master=competence_master,
            member_competence=member_competence,
            diversity_reranker=DiversityReranker()
        )

    # =========================================================
    # 推薦
    # =========================================================
    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[str] = None,
        category_filter: Optional[str] = None,
        use_diversity: bool = True,
        diversity_strategy: str = "hybrid"
    ) -> List[Recommendation]:
        """特定会員に対する推薦を生成"""
        if not self.mf_model.is_fitted:
            raise MLModelNotTrainedError()

        # コールドスタート問題のチェック：会員が学習データに存在するか
        if member_code not in self.mf_model.member_index:
            raise ColdStartError(member_code)

        # 既習得力量を取得
        acquired = self._get_acquired_competences(member_code)

        # Top-K推薦
        try:
            candidates = self.mf_model.predict_top_k(
                member_code=member_code,
                k=top_n * 3 if use_diversity else top_n,
                exclude_acquired=True,
                acquired_competences=acquired
            )
        except ValueError as e:
            # predict_top_kからのValueErrorもColdStartErrorに変換
            if "学習データに存在しません" in str(e):
                raise ColdStartError(member_code) from e
            raise

        # 力量情報を付加
        enriched = []
        for code, score in candidates:
            info = self.competence_master[self.competence_master["力量コード"] == code]
            if len(info) > 0:
                enriched.append((code, score, info.iloc[0]))

        # フィルタリング
        filtered = []
        for code, score, info in enriched:
            if competence_type and info["力量タイプ"] != competence_type:
                continue
            if category_filter:
                cat = str(info.get("力量カテゴリー名", ""))
                if category_filter.lower() not in cat.lower():
                    continue
            filtered.append((code, score))

        # 多様性再ランキング
        if use_diversity and len(filtered) > 0:
            if diversity_strategy == "mmr":
                final = self.diversity_reranker.rerank_mmr(filtered, self.competence_master, k=top_n)
            elif diversity_strategy == "category":
                final = self.diversity_reranker.rerank_category_diversity(filtered, self.competence_master, k=top_n)
            elif diversity_strategy == "type":
                final = self.diversity_reranker.rerank_type_diversity(filtered, self.competence_master, k=top_n)
            elif diversity_strategy == "hybrid":
                final = self.diversity_reranker.rerank_hybrid(filtered, self.competence_master, k=top_n)
            else:
                final = filtered[:top_n]
        else:
            final = filtered[:top_n]

        # Recommendationオブジェクトに変換
        results = []
        for code, score in final:
            info = self.competence_master[self.competence_master["力量コード"] == code].iloc[0]
            priority = self._normalize_score(score, final)
            reason = self._generate_reason(info, score, use_diversity, diversity_strategy)

            rec = Recommendation(
                competence_code=code,
                competence_name=info["力量名"],
                competence_type=info["力量タイプ"],
                category=info.get("力量カテゴリー名", ""),
                priority_score=priority,
                category_importance=0.0,
                acquisition_ease=0.0,
                popularity=0.0,
                reason=reason
            )
            results.append(rec)

        return results

    # =========================================================
    # 内部関数
    # =========================================================
    def _get_acquired_competences(self, member_code: str) -> List[str]:
        if member_code not in self._member_acquired_cache:
            acquired = self.member_competence[
                self.member_competence["メンバーコード"] == member_code
            ]["力量コード"].unique().tolist()
            self._member_acquired_cache[member_code] = acquired
        return self._member_acquired_cache[member_code]

    def _normalize_score(self, score: float, all_candidates: List[tuple]) -> float:
        if not all_candidates:
            return 5.0
        scores = [s for _, s in all_candidates]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return 5.0
        return round(((score - min_s) / (max_s - min_s)) * 10, 2)

    def _generate_reason(self, info: pd.Series, score: float, use_diversity: bool, strategy: str) -> str:
        name = info["力量名"]
        typ = info["力量タイプ"]
        cat = info.get("力量カテゴリー名", "")
        if typ == "SKILL":
            reason = f"あなたの習得傾向から、{name}の習得が推奨されます。"
        elif typ == "EDUCATION":
            reason = f"研修「{name}」の受講が、あなたの成長に適しています。"
        else:
            reason = f"資格「{name}」の取得が、あなたのスキル強化に有効です。"
        if cat:
            reason += f"（カテゴリ: {cat}）"
        if use_diversity:
            if strategy == "hybrid":
                reason += " バランスの取れた構成を考慮しました。"
            elif strategy == "mmr":
                reason += " 多様性を重視して選定されました。"
        return reason

    def calculate_diversity_metrics(self, recommendations: List[Recommendation]) -> Dict[str, float]:
        pairs = [(rec.competence_code, rec.priority_score) for rec in recommendations]
        return self.diversity_reranker.calculate_diversity_metrics(pairs, self.competence_master)
