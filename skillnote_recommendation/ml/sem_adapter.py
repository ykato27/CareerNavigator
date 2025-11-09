"""
SEM統合アダプター

既存のSEMモデル（SkillDomainSEMModel, SkillDependencySEMModel）と
新しいUnified/HierarchicalSEMEstimatorを統一インターフェースで使用可能にします。

並列運用により、両方のモデルを比較できます。
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SEMRecommendation:
    """SEM推薦結果の統一フォーマット"""
    member_code: str
    skill_code: str
    score: float
    method: str  # 'legacy', 'unified', 'hierarchical'
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class SEMComparisonResult:
    """複数SEMモデルの比較結果"""
    skill_code: str
    legacy_score: Optional[float] = None
    unified_score: Optional[float] = None
    hierarchical_score: Optional[float] = None
    ensemble_score: Optional[float] = None  # アンサンブル
    agreement_score: float = 0.0  # モデル間の一致度


class BaseSEMAdapter(ABC):
    """SEMアダプターの基底クラス"""

    @abstractmethod
    def calculate_score(self, member_code: str, skill_code: str) -> float:
        """SEMスコアを計算"""
        pass

    @abstractmethod
    def get_recommendations(
        self,
        member_code: str,
        top_k: int = 10,
    ) -> List[SEMRecommendation]:
        """推薦を取得"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """モデル名を取得"""
        pass


class LegacySEMAdapter(BaseSEMAdapter):
    """
    既存のSEMモデル用アダプター

    SkillDomainSEMModelをラップします。
    """

    def __init__(
        self,
        skill_domain_model,  # SkillDomainSEMModel
        skill_dependency_model=None,  # SkillDependencySEMModel (optional)
    ):
        """
        初期化

        Parameters:
        -----------
        skill_domain_model: SkillDomainSEMModel
            既存のスキル領域SEMモデル
        skill_dependency_model: SkillDependencySEMModel (optional)
            既存のスキル依存関係SEMモデル
        """
        self.skill_domain_model = skill_domain_model
        self.skill_dependency_model = skill_dependency_model

    def calculate_score(self, member_code: str, skill_code: str) -> float:
        """SEMスコアを計算（既存モデル）"""
        try:
            score = self.skill_domain_model.calculate_sem_score(member_code, skill_code)
            return score
        except Exception as e:
            logger.debug(f"Legacy SEM score calculation failed: {e}")
            return 0.0

    def get_recommendations(
        self,
        member_code: str,
        top_k: int = 10,
    ) -> List[SEMRecommendation]:
        """推薦を取得（既存モデル）"""
        recommendations = []

        # 全スキルに対してスコアを計算
        all_skills = self.skill_domain_model.competence_master_df['力量コード'].unique()

        scores = []
        for skill_code in all_skills:
            score = self.calculate_score(member_code, skill_code)
            if score > 0:
                scores.append((skill_code, score))

        # スコアでソート
        scores.sort(key=lambda x: x[1], reverse=True)

        # 上位k件を返す
        for skill_code, score in scores[:top_k]:
            recommendations.append(
                SEMRecommendation(
                    member_code=member_code,
                    skill_code=skill_code,
                    score=score,
                    method='legacy',
                    confidence=score,  # スコア自体を信頼度とする
                    explanation=f"既存SEMモデルによる推薦（スコア: {score:.3f}）"
                )
            )

        return recommendations

    def get_model_name(self) -> str:
        return "Legacy SEM (SkillDomainSEMModel)"


class UnifiedSEMAdapter(BaseSEMAdapter):
    """
    UnifiedSEMEstimator用アダプター
    """

    def __init__(
        self,
        unified_sem_model,  # UnifiedSEMEstimator
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
    ):
        """
        初期化

        Parameters:
        -----------
        unified_sem_model: UnifiedSEMEstimator
            統一SEM推定器
        member_competence_df: pd.DataFrame
            メンバー力量データ
        competence_master_df: pd.DataFrame
            力量マスタ
        """
        self.unified_sem_model = unified_sem_model
        self.member_competence_df = member_competence_df
        self.competence_master_df = competence_master_df

        # メンバーごとの潜在変数スコアを事前計算
        self._precompute_latent_scores()

    def _precompute_latent_scores(self):
        """潜在変数スコアを事前計算"""
        # TODO: データからメンバー×スキル行列を作成し、潜在変数スコアを計算
        # 現在は簡易実装
        self.latent_scores = {}

    def calculate_score(self, member_code: str, skill_code: str) -> float:
        """SEMスコアを計算（統一モデル）"""
        # TODO: 潜在変数スコアと構造係数から計算
        return 0.5  # 仮実装

    def get_recommendations(
        self,
        member_code: str,
        top_k: int = 10,
    ) -> List[SEMRecommendation]:
        """推薦を取得（統一モデル）"""
        # TODO: 実装
        return []

    def get_model_name(self) -> str:
        return "Unified SEM (UnifiedSEMEstimator)"


class HierarchicalSEMAdapter(BaseSEMAdapter):
    """
    HierarchicalSEMEstimator用アダプター
    """

    def __init__(
        self,
        hierarchical_sem_result,  # HierarchicalSEMResult
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
    ):
        """
        初期化

        Parameters:
        -----------
        hierarchical_sem_result: HierarchicalSEMResult
            階層的SEM推定結果
        member_competence_df: pd.DataFrame
            メンバー力量データ
        competence_master_df: pd.DataFrame
            力量マスタ
        """
        self.result = hierarchical_sem_result
        self.member_competence_df = member_competence_df
        self.competence_master_df = competence_master_df

    def calculate_score(self, member_code: str, skill_code: str) -> float:
        """SEMスコアを計算（階層モデル）"""
        # TODO: ドメインスコアから計算
        return 0.5  # 仮実装

    def get_recommendations(
        self,
        member_code: str,
        top_k: int = 10,
    ) -> List[SEMRecommendation]:
        """推薦を取得（階層モデル）"""
        # TODO: 実装
        return []

    def get_model_name(self) -> str:
        return "Hierarchical SEM (HierarchicalSEMEstimator)"


class SEMEnsemble:
    """
    複数SEMモデルのアンサンブル

    既存モデルと新モデルを並列運用し、結果を比較・統合します。
    """

    def __init__(
        self,
        adapters: List[BaseSEMAdapter],
        weights: Optional[List[float]] = None,
    ):
        """
        初期化

        Parameters:
        -----------
        adapters: List[BaseSEMAdapter]
            SEMアダプターのリスト
        weights: List[float] (optional)
            各モデルの重み（合計1.0）
        """
        self.adapters = adapters

        if weights is None:
            # デフォルト: 均等重み
            weights = [1.0 / len(adapters)] * len(adapters)

        if len(weights) != len(adapters):
            raise ValueError("重みの数とアダプターの数が一致しません")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("重みの合計は1.0である必要があります")

        self.weights = weights

        logger.info(f"SEMEnsembleを初期化: {len(adapters)}個のモデル")
        for adapter, weight in zip(adapters, weights):
            logger.info(f"  - {adapter.get_model_name()}: 重み={weight:.3f}")

    def get_ensemble_recommendations(
        self,
        member_code: str,
        top_k: int = 10,
        comparison_mode: bool = False,
    ) -> List[SEMRecommendation]:
        """
        アンサンブル推薦を取得

        Parameters:
        -----------
        member_code: str
            メンバーコード
        top_k: int
            上位k件
        comparison_mode: bool
            比較モード（各モデルの結果を個別に返す）

        Returns:
        --------
        List[SEMRecommendation]
            推薦リスト
        """
        if comparison_mode:
            # 比較モード: 各モデルの推薦を個別に返す
            all_recommendations = []
            for adapter in self.adapters:
                recs = adapter.get_recommendations(member_code, top_k)
                all_recommendations.extend(recs)
            return all_recommendations

        else:
            # アンサンブルモード: スコアを統合
            return self._ensemble_recommendations(member_code, top_k)

    def _ensemble_recommendations(
        self,
        member_code: str,
        top_k: int,
    ) -> List[SEMRecommendation]:
        """アンサンブル推薦を計算"""
        # 各モデルの推薦を取得
        all_model_recs = [
            adapter.get_recommendations(member_code, top_k * 2)  # 多めに取得
            for adapter in self.adapters
        ]

        # スキルごとにスコアを集約
        skill_scores = {}

        for model_recs, weight in zip(all_model_recs, self.weights):
            for rec in model_recs:
                if rec.skill_code not in skill_scores:
                    skill_scores[rec.skill_code] = 0.0
                skill_scores[rec.skill_code] += rec.score * weight

        # スコアでソート
        sorted_skills = sorted(
            skill_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 上位k件を返す
        recommendations = []
        for skill_code, score in sorted_skills[:top_k]:
            recommendations.append(
                SEMRecommendation(
                    member_code=member_code,
                    skill_code=skill_code,
                    score=score,
                    method='ensemble',
                    confidence=score,
                    explanation=f"アンサンブルSEMスコア: {score:.3f}"
                )
            )

        return recommendations

    def compare_models(
        self,
        member_code: str,
        skill_codes: List[str],
    ) -> pd.DataFrame:
        """
        複数モデルの予測を比較

        Parameters:
        -----------
        member_code: str
            メンバーコード
        skill_codes: List[str]
            比較対象のスキルコード

        Returns:
        --------
        pd.DataFrame
            比較結果
        """
        results = []

        for skill_code in skill_codes:
            row = {'skill_code': skill_code}

            for adapter in self.adapters:
                model_name = adapter.get_model_name()
                score = adapter.calculate_score(member_code, skill_code)
                row[model_name] = score

            # アンサンブルスコア
            ensemble_score = sum(
                row[adapter.get_model_name()] * weight
                for adapter, weight in zip(self.adapters, self.weights)
            )
            row['ensemble'] = ensemble_score

            # モデル間の一致度（標準偏差の逆数）
            scores = [row[adapter.get_model_name()] for adapter in self.adapters]
            if len(scores) > 1:
                import numpy as np
                std = np.std(scores)
                agreement = 1.0 / (1.0 + std)  # 標準偏差が小さいほど一致度が高い
            else:
                agreement = 1.0

            row['agreement'] = agreement

            results.append(row)

        return pd.DataFrame(results)

    def get_model_statistics(self) -> pd.DataFrame:
        """
        各モデルの統計情報を取得

        Returns:
        --------
        pd.DataFrame
            モデル統計
        """
        stats = []

        for adapter, weight in zip(self.adapters, self.weights):
            stats.append({
                'model': adapter.get_model_name(),
                'weight': weight,
                'type': adapter.__class__.__name__,
            })

        return pd.DataFrame(stats)
