"""
階層的SEM推定器（Hierarchical SEM Estimator）

スキル1000個に対応した階層的構造方程式モデリング:
- レベル1: 個別スキル（観測変数、1000個）
- レベル2: ドメイン力量（潜在変数、20-30個）
- レベル3: 総合力量（上位潜在変数、3-5個）

段階的推定:
1. Stage 1: スキル → ドメイン力量（各ドメインで独立したSEM）
2. Stage 2: ドメイン力量 → 総合力量（統合モデル）

並列処理により高速化（推定時間: 10-30秒）
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from skillnote_recommendation.ml.unified_sem_estimator import (
    UnifiedSEMEstimator,
    MeasurementModelSpec,
    StructuralModelSpec,
    SEMFitIndices,
    SEMParameter,
)

logger = logging.getLogger(__name__)


@dataclass
class DomainDefinition:
    """ドメイン定義"""
    domain_name: str
    skills: List[str]
    parent_domain: Optional[str] = None  # 上位ドメイン（Level 3用）
    level: int = 1  # 1: スキル→ドメイン, 2: ドメイン→総合


@dataclass
class HierarchicalSEMResult:
    """階層的SEM推定結果"""
    # ドメインレベルのモデル
    domain_models: Dict[str, UnifiedSEMEstimator]
    domain_fit_indices: Dict[str, SEMFitIndices]

    # 統合モデル
    integration_model: Optional[UnifiedSEMEstimator] = None
    integration_fit_indices: Optional[SEMFitIndices] = None

    # ドメインスコア
    domain_scores: Optional[pd.DataFrame] = None

    # 全体の適合度
    overall_fit: Optional[Dict[str, float]] = None

    # 実行時間
    elapsed_time: float = 0.0
    n_domains: int = 0
    n_skills: int = 0


class HierarchicalSEMEstimator:
    """
    階層的SEM推定器

    大規模データ（スキル1000個）に対応した階層的推定を行います。

    使用例:
    --------
    # ドメイン定義
    domains = [
        DomainDefinition('Python開発力', ['Python基礎', 'Django', ...], '技術力'),
        DomainDefinition('Web開発力', ['HTML', 'CSS', ...], '技術力'),
        DomainDefinition('要件定義力', ['要件分析', ...], 'ビジネス力'),
    ]

    # 推定
    hsem = HierarchicalSEMEstimator(domains)
    result = hsem.fit(data, n_jobs=4)  # 並列処理

    # 結果
    print(result.overall_fit)
    """

    def __init__(
        self,
        domain_definitions: List[DomainDefinition],
        confidence_level: float = 0.95,
        method: str = 'ML',
    ):
        """
        初期化

        Parameters:
        -----------
        domain_definitions: List[DomainDefinition]
            ドメイン定義のリスト
        confidence_level: float
            信頼区間のレベル
        method: str
            推定方法 ('ML' or 'GLS')
        """
        self.domain_definitions = domain_definitions
        self.confidence_level = confidence_level
        self.method = method

        # ドメイン構造の検証
        self._validate_domains()

        # 階層構造の構築
        self.level1_domains: List[DomainDefinition] = []  # スキル→ドメイン
        self.level2_domains: List[DomainDefinition] = []  # ドメイン→総合
        self._build_hierarchy()

        # 結果
        self.result: Optional[HierarchicalSEMResult] = None
        self.is_fitted = False

    def _validate_domains(self):
        """ドメイン定義の妥当性を検証"""
        domain_names = set()
        all_skills = set()

        for domain in self.domain_definitions:
            # 重複チェック
            if domain.domain_name in domain_names:
                raise ValueError(f"ドメイン名が重複しています: {domain.domain_name}")
            domain_names.add(domain.domain_name)

            # スキルの重複チェック（レベル1のみ）
            if domain.level == 1:
                for skill in domain.skills:
                    if skill in all_skills:
                        logger.warning(f"スキル {skill} が複数のドメインに存在します")
                    all_skills.add(skill)

            # スキル数のチェック
            if len(domain.skills) < 2:
                raise ValueError(f"ドメイン {domain.domain_name} のスキル数が不足しています（最低2個必要）")

    def _build_hierarchy(self):
        """階層構造を構築"""
        for domain in self.domain_definitions:
            if domain.level == 1:
                self.level1_domains.append(domain)
            elif domain.level == 2:
                self.level2_domains.append(domain)
            else:
                raise ValueError(f"未知のレベル: {domain.level}")

        logger.info(f"階層構造: Level1={len(self.level1_domains)}ドメイン, Level2={len(self.level2_domains)}ドメイン")

    def fit(
        self,
        data: pd.DataFrame,
        n_jobs: int = 1,
        use_multiprocessing: bool = False,
    ) -> HierarchicalSEMResult:
        """
        階層的SEM推定を実行

        Parameters:
        -----------
        data: pd.DataFrame
            観測データ（各列がスキル）
        n_jobs: int
            並列ジョブ数（1=逐次処理）
        use_multiprocessing: bool
            マルチプロセス使用（Trueで高速化、メモリ使用量増）

        Returns:
        --------
        HierarchicalSEMResult
            推定結果
        """
        import time
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("階層的SEM推定を開始")
        logger.info(f"データサイズ: n={len(data)}, スキル数={len(data.columns)}")
        logger.info(f"並列ジョブ数: {n_jobs}")
        logger.info("=" * 70)

        # Stage 1: スキル → ドメイン力量
        logger.info("\n【Stage 1】スキル → ドメイン力量")
        domain_models, domain_fit_indices = self._fit_domain_models(
            data, n_jobs, use_multiprocessing
        )

        # ドメインスコアを計算
        logger.info("\nドメインスコアを計算中...")
        domain_scores = self._compute_domain_scores(data, domain_models)

        # Stage 2: ドメイン力量 → 総合力量（Level 2が定義されている場合）
        integration_model = None
        integration_fit_indices = None

        if self.level2_domains:
            logger.info("\n【Stage 2】ドメイン力量 → 総合力量")
            integration_model, integration_fit_indices = self._fit_integration_model(
                domain_scores
            )

        # 全体の適合度を計算
        overall_fit = self._compute_overall_fit(domain_fit_indices, integration_fit_indices)

        elapsed_time = time.time() - start_time

        # 結果を保存
        self.result = HierarchicalSEMResult(
            domain_models=domain_models,
            domain_fit_indices=domain_fit_indices,
            integration_model=integration_model,
            integration_fit_indices=integration_fit_indices,
            domain_scores=domain_scores,
            overall_fit=overall_fit,
            elapsed_time=elapsed_time,
            n_domains=len(domain_models),
            n_skills=len(data.columns),
        )

        self.is_fitted = True

        logger.info("\n" + "=" * 70)
        logger.info(f"階層的SEM推定完了（実行時間: {elapsed_time:.2f}秒）")
        logger.info(f"全体適合度: RMSEA={overall_fit['rmsea']:.3f}, CFI={overall_fit['cfi']:.3f}")
        logger.info("=" * 70)

        return self.result

    def _fit_domain_models(
        self,
        data: pd.DataFrame,
        n_jobs: int,
        use_multiprocessing: bool,
    ) -> Tuple[Dict[str, UnifiedSEMEstimator], Dict[str, SEMFitIndices]]:
        """
        ドメインレベルのSEMモデルを推定

        各ドメインで独立したSEMを推定します。
        """
        domain_models = {}
        domain_fit_indices = {}

        if n_jobs == 1:
            # 逐次処理
            for i, domain in enumerate(self.level1_domains, 1):
                logger.info(f"  [{i}/{len(self.level1_domains)}] {domain.domain_name} を推定中...")
                model, fit = self._fit_single_domain(data, domain)
                domain_models[domain.domain_name] = model
                domain_fit_indices[domain.domain_name] = fit
                logger.info(f"      ✅ 完了 (RMSEA={fit.rmsea:.3f}, CFI={fit.cfi:.3f})")

        else:
            # 並列処理
            logger.info(f"  並列処理で{len(self.level1_domains)}ドメインを推定中...")

            # スレッドプールまたはプロセスプールを選択
            if use_multiprocessing:
                # NOTE: マルチプロセスはPickle可能なオブジェクトが必要
                # UnifiedSEMEstimatorがPickle可能か確認が必要
                executor_class = ProcessPoolExecutor
                max_workers = min(n_jobs, mp.cpu_count())
            else:
                executor_class = ThreadPoolExecutor
                max_workers = n_jobs

            with executor_class(max_workers=max_workers) as executor:
                # ジョブを投入
                future_to_domain = {
                    executor.submit(self._fit_single_domain, data, domain): domain
                    for domain in self.level1_domains
                }

                # 結果を取得
                for future in as_completed(future_to_domain):
                    domain = future_to_domain[future]
                    try:
                        model, fit = future.result()
                        domain_models[domain.domain_name] = model
                        domain_fit_indices[domain.domain_name] = fit
                        logger.info(f"  ✅ {domain.domain_name} 完了 (RMSEA={fit.rmsea:.3f}, CFI={fit.cfi:.3f})")
                    except Exception as e:
                        logger.error(f"  ❌ {domain.domain_name} でエラー: {e}")
                        raise

        return domain_models, domain_fit_indices

    def _fit_single_domain(
        self,
        data: pd.DataFrame,
        domain: DomainDefinition,
    ) -> Tuple[UnifiedSEMEstimator, SEMFitIndices]:
        """
        単一ドメインのSEMを推定

        Parameters:
        -----------
        data: pd.DataFrame
            観測データ
        domain: DomainDefinition
            ドメイン定義

        Returns:
        --------
        model: UnifiedSEMEstimator
            推定済みモデル
        fit_indices: SEMFitIndices
            適合度指標
        """
        # データに存在するスキルのみを使用
        available_skills = [s for s in domain.skills if s in data.columns]

        if len(available_skills) < 2:
            raise ValueError(
                f"ドメイン {domain.domain_name} に十分なスキルがありません "
                f"(必要: 2個以上, 利用可能: {len(available_skills)}個)"
            )

        # 測定モデル仕様
        measurement = [
            MeasurementModelSpec(
                domain.domain_name,
                available_skills,
                reference_indicator=available_skills[0]  # 最初のスキルを参照指標に
            )
        ]

        # 構造モデルなし（測定モデルのみ）
        structural = []

        # SEM推定
        sem = UnifiedSEMEstimator(
            measurement,
            structural,
            method=self.method,
            confidence_level=self.confidence_level,
        )

        # データのサブセットで推定
        sem.fit(data[available_skills])

        return sem, sem.fit_indices

    def _compute_domain_scores(
        self,
        data: pd.DataFrame,
        domain_models: Dict[str, UnifiedSEMEstimator],
    ) -> pd.DataFrame:
        """
        各メンバーのドメインスコアを計算

        Parameters:
        -----------
        data: pd.DataFrame
            観測データ
        domain_models: Dict[str, UnifiedSEMEstimator]
            ドメインモデル

        Returns:
        --------
        pd.DataFrame
            ドメインスコア（各列がドメイン）
        """
        domain_scores = {}

        for domain_name, model in domain_models.items():
            # ドメインに含まれるスキルを取得
            available_skills = [s for s in model.observed_vars if s in data.columns]

            # 潜在変数スコアを予測
            scores = model.predict_latent_scores(data[available_skills])

            # ドメイン名の列を取得（通常1列）
            if domain_name in scores.columns:
                domain_scores[domain_name] = scores[domain_name]
            else:
                # 列名が異なる場合は最初の列を使用
                domain_scores[domain_name] = scores.iloc[:, 0]

        return pd.DataFrame(domain_scores, index=data.index)

    def _fit_integration_model(
        self,
        domain_scores: pd.DataFrame,
    ) -> Tuple[UnifiedSEMEstimator, SEMFitIndices]:
        """
        統合モデル（ドメイン力量 → 総合力量）を推定

        Parameters:
        -----------
        domain_scores: pd.DataFrame
            ドメインスコア

        Returns:
        --------
        model: UnifiedSEMEstimator
            統合モデル
        fit_indices: SEMFitIndices
            適合度指標
        """
        # 測定モデル仕様（各総合力量に対するドメイン）
        measurement = []
        for level2_domain in self.level2_domains:
            # このlevel2ドメインに属するlevel1ドメインを探す
            child_domains = [
                d.domain_name for d in self.level1_domains
                if d.parent_domain == level2_domain.domain_name
            ]

            if len(child_domains) < 2:
                logger.warning(
                    f"総合力量 {level2_domain.domain_name} に十分な子ドメインがありません "
                    f"({len(child_domains)}個)"
                )
                continue

            # 利用可能なドメインのみ使用
            available_domains = [d for d in child_domains if d in domain_scores.columns]

            if len(available_domains) >= 2:
                measurement.append(
                    MeasurementModelSpec(
                        level2_domain.domain_name,
                        available_domains,
                        reference_indicator=available_domains[0]
                    )
                )

        if not measurement:
            logger.warning("統合モデルの測定仕様が空です")
            return None, None

        # 構造モデル（総合力量間の関係）
        structural = []
        # TODO: ユーザー定義の構造パスを追加可能に

        # SEM推定
        sem = UnifiedSEMEstimator(
            measurement,
            structural,
            method=self.method,
            confidence_level=self.confidence_level,
        )

        # 必要なドメインスコアのみで推定
        all_required_domains = []
        for spec in measurement:
            all_required_domains.extend(spec.observed_vars)

        sem.fit(domain_scores[all_required_domains])

        return sem, sem.fit_indices

    def _compute_overall_fit(
        self,
        domain_fit_indices: Dict[str, SEMFitIndices],
        integration_fit_indices: Optional[SEMFitIndices],
    ) -> Dict[str, float]:
        """
        全体の適合度を計算

        ドメインモデルと統合モデルの適合度を平均します。
        """
        # ドメインモデルの平均適合度
        domain_rmsea = np.mean([fit.rmsea for fit in domain_fit_indices.values()])
        domain_cfi = np.mean([fit.cfi for fit in domain_fit_indices.values()])
        domain_tli = np.mean([fit.tli for fit in domain_fit_indices.values()])

        # 統合モデルの適合度
        if integration_fit_indices:
            integration_rmsea = integration_fit_indices.rmsea
            integration_cfi = integration_fit_indices.cfi
            integration_tli = integration_fit_indices.tli

            # 加重平均（ドメイン70%, 統合30%）
            overall_rmsea = 0.7 * domain_rmsea + 0.3 * integration_rmsea
            overall_cfi = 0.7 * domain_cfi + 0.3 * integration_cfi
            overall_tli = 0.7 * domain_tli + 0.3 * integration_tli
        else:
            overall_rmsea = domain_rmsea
            overall_cfi = domain_cfi
            overall_tli = domain_tli

        return {
            'rmsea': overall_rmsea,
            'cfi': overall_cfi,
            'tli': overall_tli,
            'domain_rmsea': domain_rmsea,
            'domain_cfi': domain_cfi,
            'domain_tli': domain_tli,
        }

    def predict_all_scores(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        すべてのレベルのスコアを予測

        Parameters:
        -----------
        data: pd.DataFrame
            観測データ

        Returns:
        --------
        Dict[str, pd.DataFrame]
            'domain_scores': ドメインスコア
            'total_scores': 総合力量スコア（統合モデルがある場合）
        """
        if not self.is_fitted:
            raise ValueError("モデルがまだ推定されていません。fit()を先に実行してください。")

        # ドメインスコア
        domain_scores = self._compute_domain_scores(data, self.result.domain_models)

        result = {'domain_scores': domain_scores}

        # 総合力量スコア
        if self.result.integration_model:
            required_domains = self.result.integration_model.observed_vars
            total_scores = self.result.integration_model.predict_latent_scores(
                domain_scores[required_domains]
            )
            result['total_scores'] = total_scores

        return result

    def get_skill_to_domain_loadings(self) -> pd.DataFrame:
        """
        スキル→ドメインのファクターローディングを取得

        Returns:
        --------
        pd.DataFrame
            columns: skill, domain, loading
        """
        if not self.is_fitted:
            raise ValueError("モデルがまだ推定されていません。fit()を先に実行してください。")

        loadings = []

        for domain_name, model in self.result.domain_models.items():
            for i, skill in enumerate(model.observed_vars):
                for j, latent in enumerate(model.latent_vars):
                    loading = model.Lambda[i, j]
                    if abs(loading) > 1e-6:
                        loadings.append({
                            'skill': skill,
                            'domain': domain_name,
                            'loading': loading,
                        })

        return pd.DataFrame(loadings)

    def get_domain_relationships(self) -> pd.DataFrame:
        """
        ドメイン間の関係性を取得（統合モデルから）

        Returns:
        --------
        pd.DataFrame
            ドメイン間の構造係数
        """
        if not self.is_fitted:
            raise ValueError("モデルがまだ推定されていません。fit()を先に実行してください。")

        if not self.result.integration_model:
            logger.warning("統合モデルが存在しません")
            return pd.DataFrame()

        return self.result.integration_model.get_skill_relationships()


def auto_detect_domains(
    data: pd.DataFrame,
    skill_categories: pd.DataFrame,
    n_domains: int = 20,
) -> List[DomainDefinition]:
    """
    スキルカテゴリから自動的にドメイン定義を生成

    Parameters:
    -----------
    data: pd.DataFrame
        観測データ
    skill_categories: pd.DataFrame
        スキルカテゴリ情報
        columns: ['skill_code', 'skill_name', 'category', 'parent_category']
    n_domains: int
        目標ドメイン数

    Returns:
    --------
    List[DomainDefinition]
        ドメイン定義のリスト
    """
    domains = []

    # カテゴリごとにグループ化
    category_groups = skill_categories.groupby('category')

    for category, group in category_groups:
        skills = group['skill_code'].tolist()

        # データに存在するスキルのみ
        available_skills = [s for s in skills if s in data.columns]

        if len(available_skills) >= 2:
            # 親カテゴリを取得（Level 2用）
            parent = group['parent_category'].iloc[0] if 'parent_category' in group.columns else None

            domains.append(
                DomainDefinition(
                    domain_name=category,
                    skills=available_skills,
                    parent_domain=parent,
                    level=1
                )
            )

    logger.info(f"自動検出: {len(domains)}個のドメインを生成しました")

    return domains
