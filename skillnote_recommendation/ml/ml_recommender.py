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
from skillnote_recommendation.core.reference_persons import ReferencePersonFinder


class MLRecommender:
    """機械学習ベース推薦エンジン（NMF版）"""

    def __init__(
        self,
        mf_model: MatrixFactorizationModel,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        member_master: pd.DataFrame,
        diversity_reranker: Optional[DiversityReranker] = None,
        reference_person_finder: Optional[ReferencePersonFinder] = None
    ):
        self.mf_model = mf_model
        self.competence_master = competence_master
        self.member_competence = member_competence
        self.member_master = member_master
        self.diversity_reranker = diversity_reranker or DiversityReranker()
        self.reference_person_finder = reference_person_finder
        self._member_acquired_cache = {}

    # =========================================================
    # 学習
    # =========================================================
    @classmethod
    def build(
        cls,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        member_master: pd.DataFrame
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

        # 参考人物検索エンジンを初期化
        reference_finder = ReferencePersonFinder(
            member_competence=member_competence,
            member_master=member_master,
            competence_master=competence_master
        )

        return cls(
            mf_model=mf_model,
            competence_master=competence_master,
            member_competence=member_competence,
            member_master=member_master,
            diversity_reranker=DiversityReranker(),
            reference_person_finder=reference_finder
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

            # 参考人物を検索
            reference_persons = []
            if self.reference_person_finder:
                reference_persons = self.reference_person_finder.find_reference_persons(
                    target_member_code=member_code,
                    recommended_competence_code=code,
                    top_n=3
                )

            # リッチな推薦理由を生成（参考人物情報を含む）
            reason = self._generate_rich_reason(
                member_code=member_code,
                competence_info=info,
                score=score,
                use_diversity=use_diversity,
                diversity_strategy=diversity_strategy,
                reference_persons=reference_persons
            )

            rec = Recommendation(
                competence_code=code,
                competence_name=info["力量名"],
                competence_type=info["力量タイプ"],
                category=info.get("力量カテゴリー名", ""),
                priority_score=priority,
                category_importance=0.0,
                acquisition_ease=0.0,
                popularity=0.0,
                reason=reason,
                reference_persons=reference_persons
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

    def _generate_rich_reason(self, member_code: str, competence_info: pd.Series,
                               score: float, use_diversity: bool, diversity_strategy: str,
                               reference_persons: list) -> str:
        """
        個人の力量プロファイルに基づいたリッチな推薦理由を生成

        Args:
            member_code: 対象会員コード
            competence_info: 推薦力量の情報
            score: 推薦スコア
            use_diversity: 多様性を考慮するか
            diversity_strategy: 多様性戦略
            reference_persons: 参考人物リスト

        Returns:
            リッチな推薦理由（マークダウン形式）
        """
        name = competence_info["力量名"]
        typ = competence_info["力量タイプ"]
        cat = competence_info.get("力量カテゴリー名", "")

        # 会員の力量プロファイルを分析
        acquired = self._get_acquired_competences(member_code)
        acquired_count = len(acquired)

        # カテゴリ別の保有力量を分析
        category_profile = self._analyze_category_profile(member_code)

        # === 推薦理由の構築 ===
        reason_parts = []

        # 1. 導入部：なぜこの力量が推薦されるのか
        intro = self._generate_reason_intro(name, typ, cat, score, acquired_count, category_profile)
        reason_parts.append(intro)

        # 2. 個人プロファイルとの関連性
        profile_relevance = self._generate_profile_relevance(
            member_code, name, typ, cat, acquired, category_profile
        )
        if profile_relevance:
            reason_parts.append(profile_relevance)

        # 3. 多様性戦略の説明
        if use_diversity:
            diversity_explanation = self._generate_diversity_explanation(diversity_strategy)
            if diversity_explanation:
                reason_parts.append(diversity_explanation)

        # 4. 習得によるメリット
        benefits = self._generate_benefits(typ, cat, category_profile)
        if benefits:
            reason_parts.append(benefits)

        return "\n\n".join(reason_parts)

    def _generate_reason_intro(self, name: str, typ: str, cat: str,
                                score: float, acquired_count: int,
                                category_profile: Dict) -> str:
        """推薦理由の導入部を生成"""
        score_pct = int(score * 100) if score <= 1 else int(score * 10)

        if typ == "SKILL":
            intro = (
                f"**スキル「{name}」**は、あなたの現在の力量プロファイル（保有力量{acquired_count}個）と"
                f"機械学習モデルの分析から、**適合度{score_pct}%**で推薦されます。"
            )
        elif typ == "EDUCATION":
            intro = (
                f"**研修「{name}」**は、あなたのキャリアパス分析と保有力量{acquired_count}個の傾向から、"
                f"**適合度{score_pct}%**で受講が推奨されます。"
            )
        else:
            intro = (
                f"**資格「{name}」**は、あなたの現在のスキルセット（{acquired_count}個の力量）を考慮し、"
                f"**適合度{score_pct}%**で取得が推奨されます。"
            )

        if cat:
            intro += f"\n\n📁 **カテゴリ**: {cat}"

        return intro

    def _generate_profile_relevance(self, member_code: str, name: str, typ: str,
                                     cat: str, acquired: List[str],
                                     category_profile: Dict) -> str:
        """個人プロファイルとの関連性を説明"""
        if not category_profile:
            return ""

        # カテゴリ別の保有状況を分析
        if cat and cat in category_profile:
            cat_count = category_profile[cat]
            total_cat = len(self.competence_master[
                self.competence_master["力量カテゴリー名"] == cat
            ])
            cat_ratio = int((cat_count / total_cat * 100)) if total_cat > 0 else 0

            relevance = (
                f"🎯 **あなたのプロファイルとの関連性**\n\n"
                f"あなたは既に「{cat}」カテゴリの力量を{cat_count}個保有しており、"
                f"このカテゴリの{cat_ratio}%をカバーしています。\n"
                f"この力量を習得することで、{cat}分野での専門性がさらに強化されます。"
            )
            return relevance

        # カテゴリ情報がない場合は、全体的な傾向を説明
        top_category = max(category_profile.items(), key=lambda x: x[1])[0]
        top_count = category_profile[top_category]

        relevance = (
            f"🎯 **あなたのプロファイルとの関連性**\n\n"
            f"あなたの主要な力量は「{top_category}」カテゴリ（{top_count}個）です。\n"
            f"「{name}」を習得することで、キャリアの幅を広げることができます。"
        )
        return relevance

    def _generate_diversity_explanation(self, strategy: str) -> str:
        """多様性戦略の説明を生成"""
        if strategy == "hybrid":
            return (
                "⚖️ **推薦戦略**: バランス重視\n\n"
                "類似性と多様性のバランスを考慮し、あなたのキャリアに最適な構成を提案しています。"
            )
        elif strategy == "mmr":
            return (
                "🎨 **推薦戦略**: 多様性重視\n\n"
                "既存の力量と重複を避け、新しい分野への挑戦を重視した推薦です。"
            )
        elif strategy == "category":
            return (
                "📚 **推薦戦略**: カテゴリ多様性\n\n"
                "異なるカテゴリの力量をバランスよく推薦し、幅広いスキルセットの構築を支援します。"
            )
        elif strategy == "type":
            return (
                "🔄 **推薦戦略**: タイプ多様性\n\n"
                "スキル・研修・資格のバランスを考慮し、総合的な成長を目指します。"
            )
        return ""

    def _generate_benefits(self, typ: str, cat: str, category_profile: Dict) -> str:
        """習得によるメリットを生成"""
        if typ == "SKILL":
            benefits = (
                "✨ **習得によるメリット**\n\n"
                "- 実務での即戦力スキルとして活用可能\n"
                "- 同様のスキルを持つメンバーとの協業機会が増加\n"
                "- キャリアの選択肢が広がります"
            )
        elif typ == "EDUCATION":
            benefits = (
                "✨ **受講によるメリット**\n\n"
                "- 体系的な知識習得が可能\n"
                "- 研修を通じた社内ネットワーク構築\n"
                "- 認定取得によるスキルの証明"
            )
        else:
            benefits = (
                "✨ **取得によるメリット**\n\n"
                "- 外部に対するスキル証明が可能\n"
                "- キャリアアップの機会増加\n"
                "- 専門性の客観的な評価"
            )

        return benefits

    def _analyze_category_profile(self, member_code: str) -> Dict[str, int]:
        """会員のカテゴリ別力量保有状況を分析"""
        member_comps = self.member_competence[
            self.member_competence["メンバーコード"] == member_code
        ]

        # 力量コードからカテゴリ情報を取得
        merged = member_comps.merge(
            self.competence_master[["力量コード", "力量カテゴリー名"]],
            on="力量コード",
            how="left"
        )

        # カテゴリ別にカウント
        category_counts = merged["力量カテゴリー名"].value_counts().to_dict()

        return category_counts

    def calculate_diversity_metrics(self, recommendations: List[Recommendation]) -> Dict[str, float]:
        pairs = [(rec.competence_code, rec.priority_score) for rec in recommendations]
        return self.diversity_reranker.calculate_diversity_metrics(pairs, self.competence_master)
