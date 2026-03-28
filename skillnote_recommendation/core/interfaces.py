"""
インターフェース定義

GAFAレベルの設計:
- Protocol（構造的部分型）
- ABC（抽象基底クラス）
- 依存性注入の基盤
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any, runtime_checkable
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from skillnote_recommendation.core.models import Recommendation


# =============================================================================
# Matrix Factorization関連インターフェース
# =============================================================================


@runtime_checkable
class MatrixFactorizationProtocol(Protocol):
    """
    Matrix Factorizationモデルのインターフェース

    構造的部分型（Protocol）を使用することで、
    ダックタイピングでインターフェースを満たすことができる
    """

    is_fitted: bool
    member_codes: list[str] | None
    competence_codes: list[str] | None

    def fit(self, skill_matrix: pd.DataFrame) -> "MatrixFactorizationProtocol":
        """モデルを学習"""
        ...

    def predict(self, member_code: str, competence_codes: list[str] | None = None) -> pd.Series:
        """特定メンバーに対する力量のスコアを予測"""
        ...

    def predict_top_k(
        self,
        member_code: str,
        k: int = 10,
        exclude_acquired: bool = True,
        acquired_competences: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-K推薦を生成"""
        ...

    def get_member_factors(self, member_code: str) -> NDArray[np.float64]:
        """メンバーの潜在因子ベクトルを取得"""
        ...

    def get_competence_factors(self, competence_code: str) -> NDArray[np.float64]:
        """力量の潜在因子ベクトルを取得"""
        ...

    def save(self, filepath: str) -> None:
        """モデルを保存"""
        ...


# =============================================================================
# Diversity Reranker関連インターフェース
# =============================================================================


@runtime_checkable
class DiversityRerankerProtocol(Protocol):
    """多様性再ランキングのインターフェース"""

    def rerank_mmr(
        self,
        candidates: list[tuple[str, float]],
        competence_info: pd.DataFrame,
        k: int = 10,
        use_position_aware: bool = False,
    ) -> list[tuple[str, float]]:
        """MMR（Maximal Marginal Relevance）による再ランキング"""
        ...

    def rerank_category_diversity(
        self,
        candidates: list[tuple[str, float]],
        competence_info: pd.DataFrame,
        k: int = 10,
        max_per_category: int | None = None,
    ) -> list[tuple[str, float]]:
        """カテゴリ多様性を考慮した再ランキング"""
        ...

    def rerank_type_diversity(
        self,
        candidates: list[tuple[str, float]],
        competence_info: pd.DataFrame,
        k: int = 10,
        type_ratios: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """タイプ多様性を考慮した再ランキング"""
        ...

    def rerank_hybrid(
        self,
        candidates: list[tuple[str, float]],
        competence_info: pd.DataFrame,
        k: int = 10,
        max_per_category: int | None = 3,
        type_ratios: dict[str, float] | None = None,
        use_position_aware: bool = True,
    ) -> list[tuple[str, float]]:
        """ハイブリッド再ランキング"""
        ...

    def calculate_diversity_metrics(
        self, recommendations: list[tuple[str, float]], competence_info: pd.DataFrame
    ) -> dict[str, float]:
        """推薦結果の多様性指標を計算"""
        ...


# =============================================================================
# Reference Person Finder関連インターフェース
# =============================================================================


@runtime_checkable
class ReferencePersonFinderProtocol(Protocol):
    """参考人物検索のインターフェース"""

    def find_reference_persons(
        self, target_member_code: str, recommended_competence_code: str, top_n: int = 3
    ) -> list[Any]:
        """参考人物を検索"""
        ...


# =============================================================================
# Recommender関連インターフェース
# =============================================================================


class BaseRecommender(ABC):
    """
    推薦エンジンの抽象基底クラス

    全ての推薦エンジンはこのクラスを継承する
    """

    @abstractmethod
    def recommend(self, member_code: str, top_n: int = 10, **kwargs: Any) -> list[Recommendation]:
        """
        推薦を生成

        Args:
            member_code: メンバーコード
            top_n: 推薦数
            **kwargs: 追加パラメータ

        Returns:
            推薦結果のリスト

        Raises:
            ColdStartError: コールドスタート問題の場合
            ModelNotTrainedError: モデルが未学習の場合
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """推薦可能な状態かどうか"""
        pass


# =============================================================================
# Data Preprocessor関連インターフェース
# =============================================================================


@runtime_checkable
class DataPreprocessorProtocol(Protocol):
    """データ前処理のインターフェース"""

    def preprocess(
        self, skill_matrix: pd.DataFrame, verbose: bool = True
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        スキルマトリックスを前処理

        Args:
            skill_matrix: スキルマトリックス
            verbose: 詳細情報を出力するか

        Returns:
            (前処理済みマトリックス, 統計情報)
        """
        ...


# =============================================================================
# Evaluator関連インターフェース
# =============================================================================


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """評価のインターフェース"""

    def evaluate_recommendations(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        top_k: int = 10,
        member_sample: list[str] | None = None,
        similarity_data: pd.DataFrame | None = None,
        include_extended_metrics: bool = True,
    ) -> dict[str, float]:
        """推薦結果を評価"""
        ...

    def calculate_diversity_metrics(
        self,
        recommendations_list: list[list[Recommendation]],
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame | None = None,
        include_advanced_metrics: bool = True,
    ) -> dict[str, float]:
        """多様性指標を計算"""
        ...


# =============================================================================
# Hyperparameter Tuner関連インターフェース
# =============================================================================


class BaseHyperparameterTuner(ABC):
    """ハイパーパラメータチューナーの抽象基底クラス"""

    @abstractmethod
    def optimize(
        self, show_progress_bar: bool = True, callbacks: list[Any] | None = None
    ) -> tuple[dict[str, Any], float]:
        """
        ハイパーパラメータ最適化を実行

        Returns:
            (最適パラメータ, 最小評価値)
        """
        pass

    @abstractmethod
    def get_best_model(self) -> Any:
        """最適なパラメータで学習したモデルを取得"""
        pass

    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """最適化の履歴を取得"""
        pass
