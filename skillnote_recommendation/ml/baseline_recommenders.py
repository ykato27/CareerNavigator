"""
ベースライン推薦モデル

機械学習ベース推薦システムの性能を評価するための比較対象となるベースラインモデル。

実装モデル:
1. RandomRecommender: ランダム推薦
2. PopularityRecommender: 人気度ベース推薦
3. CategoryBasedRecommender: カテゴリベース推薦
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from collections import Counter
from skillnote_recommendation.core.models import Recommendation


class BaselineRecommender:
    """ベースライン推薦モデルの基底クラス"""

    def __init__(
        self,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        member_master: pd.DataFrame,
    ):
        """
        初期化

        Args:
            competence_master: 力量マスタ
            member_competence: メンバー習得力量データ
            member_master: メンバーマスタ
        """
        self.competence_master = competence_master
        self.member_competence = member_competence
        self.member_master = member_master
        self._member_acquired_cache = {}

    def _get_acquired_competences(self, member_code: str) -> List[str]:
        """メンバーが既に習得している力量を取得"""
        if member_code not in self._member_acquired_cache:
            acquired = (
                self.member_competence[self.member_competence["メンバーコード"] == member_code][
                    "力量コード"
                ]
                .unique()
                .tolist()
            )
            self._member_acquired_cache[member_code] = acquired
        return self._member_acquired_cache[member_code]

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Recommendation]:
        """
        推薦を生成（サブクラスで実装）

        Args:
            member_code: 対象メンバーコード
            top_n: 推薦数
            competence_type: 力量タイプフィルタ
            category_filter: カテゴリフィルタ

        Returns:
            推薦リスト
        """
        raise NotImplementedError("Subclass must implement recommend()")


class RandomRecommender(BaselineRecommender):
    """ランダム推薦モデル

    既習得力量を除いた全力量からランダムに推薦する最も単純なベースライン。
    """

    def __init__(
        self,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        member_master: pd.DataFrame,
        random_state: int = 42,
    ):
        super().__init__(competence_master, member_competence, member_master)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Recommendation]:
        """ランダムに推薦を生成"""
        # 既習得力量を取得
        acquired = self._get_acquired_competences(member_code)

        # 候補力量を抽出（既習得を除く）
        candidates = self.competence_master[
            ~self.competence_master["力量コード"].isin(acquired)
        ].copy()

        # フィルタリング
        if competence_type:
            candidates = candidates[candidates["力量タイプ"].isin(competence_type)]

        if category_filter:
            candidates = candidates[
                candidates["力量カテゴリー名"].str.contains(category_filter, case=False, na=False)
            ]

        # ランダムに選択
        if len(candidates) == 0:
            return []

        n_select = min(top_n, len(candidates))
        selected_indices = self.rng.choice(len(candidates), size=n_select, replace=False)
        selected = candidates.iloc[selected_indices]

        # Recommendationオブジェクトに変換
        results = []
        for _, row in selected.iterrows():
            rec = Recommendation(
                competence_code=row["力量コード"],
                competence_name=row["力量名"],
                competence_type=row["力量タイプ"],
                category=row.get("力量カテゴリー名", ""),
                priority_score=5.0,  # 一律5.0
                category_importance=0.0,
                acquisition_ease=0.0,
                popularity=0.0,
                reason="ランダムに選択された力量です。",
                reference_persons=[],
            )
            results.append(rec)

        return results


class PopularityRecommender(BaselineRecommender):
    """人気度ベース推薦モデル

    最も多くのメンバーが習得している力量を推薦する。
    協調フィルタリングの最も単純な形。
    """

    def __init__(
        self,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        member_master: pd.DataFrame,
    ):
        super().__init__(competence_master, member_competence, member_master)
        self._compute_popularity()

    def _compute_popularity(self):
        """各力量の人気度（習得者数）を計算"""
        popularity_counts = (
            self.member_competence["力量コード"].value_counts().to_dict()
        )

        # 力量マスタに人気度スコアを追加
        self.competence_master = self.competence_master.copy()
        self.competence_master["人気度スコア"] = self.competence_master["力量コード"].map(
            popularity_counts
        ).fillna(0)

        # 正規化（0-10スケール）
        max_count = self.competence_master["人気度スコア"].max()
        if max_count > 0:
            self.competence_master["人気度スコア"] = (
                self.competence_master["人気度スコア"] / max_count * 10
            )

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Recommendation]:
        """人気度順に推薦を生成"""
        # 既習得力量を取得
        acquired = self._get_acquired_competences(member_code)

        # 候補力量を抽出（既習得を除く）
        candidates = self.competence_master[
            ~self.competence_master["力量コード"].isin(acquired)
        ].copy()

        # フィルタリング
        if competence_type:
            candidates = candidates[candidates["力量タイプ"].isin(competence_type)]

        if category_filter:
            candidates = candidates[
                candidates["力量カテゴリー名"].str.contains(category_filter, case=False, na=False)
            ]

        # 人気度順にソート
        candidates = candidates.sort_values("人気度スコア", ascending=False)

        # Top-Nを選択
        selected = candidates.head(top_n)

        # Recommendationオブジェクトに変換
        results = []
        for _, row in selected.iterrows():
            popularity_score = row["人気度スコア"]
            rec = Recommendation(
                competence_code=row["力量コード"],
                competence_name=row["力量名"],
                competence_type=row["力量タイプ"],
                category=row.get("力量カテゴリー名", ""),
                priority_score=popularity_score,
                category_importance=0.0,
                acquisition_ease=0.0,
                popularity=popularity_score / 10.0,  # 0-1スケール
                reason=f"多くのメンバーが習得している人気の力量です（習得率: {int(popularity_score * 10)}%）。",
                reference_persons=[],
            )
            results.append(rec)

        return results


class CategoryBasedRecommender(BaselineRecommender):
    """カテゴリベース推薦モデル

    メンバーが既に保有している力量のカテゴリから、
    同じカテゴリの未習得力量を推薦する。
    """

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Recommendation]:
        """カテゴリベースで推薦を生成"""
        # 既習得力量を取得
        acquired = self._get_acquired_competences(member_code)

        # メンバーの保有カテゴリを分析
        member_comps = self.member_competence[
            self.member_competence["メンバーコード"] == member_code
        ]

        # 力量コードから力量カテゴリーを取得
        merged = member_comps.merge(
            self.competence_master[["力量コード", "力量カテゴリー名"]],
            on="力量コード",
            how="left",
        )

        # カテゴリ別の保有数をカウント
        category_counts = merged["力量カテゴリー名"].value_counts().to_dict()

        if not category_counts:
            # カテゴリ情報がない場合はランダム推薦にフォールバック
            random_rec = RandomRecommender(
                self.competence_master, self.member_competence, self.member_master
            )
            return random_rec.recommend(member_code, top_n, competence_type, category_filter)

        # 候補力量を抽出（既習得を除く）
        candidates = self.competence_master[
            ~self.competence_master["力量コード"].isin(acquired)
        ].copy()

        # フィルタリング
        if competence_type:
            candidates = candidates[candidates["力量タイプ"].isin(competence_type)]

        if category_filter:
            candidates = candidates[
                candidates["力量カテゴリー名"].str.contains(category_filter, case=False, na=False)
            ]

        # カテゴリスコアを計算（保有数が多いカテゴリほど高スコア）
        candidates["カテゴリスコア"] = candidates["力量カテゴリー名"].map(category_counts).fillna(0)

        # カテゴリスコア順にソート
        candidates = candidates.sort_values("カテゴリスコア", ascending=False)

        # Top-Nを選択
        selected = candidates.head(top_n)

        # Recommendationオブジェクトに変換
        results = []
        max_score = selected["カテゴリスコア"].max() if len(selected) > 0 else 1
        for _, row in selected.iterrows():
            category_score = row["カテゴリスコア"]
            normalized_score = (category_score / max_score * 10) if max_score > 0 else 5.0

            rec = Recommendation(
                competence_code=row["力量コード"],
                competence_name=row["力量名"],
                competence_type=row["力量タイプ"],
                category=row.get("力量カテゴリー名", ""),
                priority_score=normalized_score,
                category_importance=category_score / (max_score if max_score > 0 else 1),
                acquisition_ease=0.0,
                popularity=0.0,
                reason=f"あなたが保有している「{row['力量カテゴリー名']}」カテゴリの力量です（保有数: {int(category_score)}個）。",
                reference_persons=[],
            )
            results.append(rec)

        return results
