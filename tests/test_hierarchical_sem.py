"""
HierarchicalSEMEstimatorの動作確認テスト
"""

import sys
sys.path.insert(0, '/home/user/CareerNavigator')

import numpy as np
import pandas as pd
import time

# 直接ファイルをimport
import importlib.util

# UnifiedSEMEstimatorをimport
spec_unified = importlib.util.spec_from_file_location(
    "unified_sem_estimator",
    "/home/user/CareerNavigator/skillnote_recommendation/ml/unified_sem_estimator.py"
)
unified_sem_module = importlib.util.module_from_spec(spec_unified)
spec_unified.loader.exec_module(unified_sem_module)

# HierarchicalSEMEstimatorをimport
spec_hierarchical = importlib.util.spec_from_file_location(
    "hierarchical_sem_estimator",
    "/home/user/CareerNavigator/skillnote_recommendation/ml/hierarchical_sem_estimator.py"
)
hierarchical_sem_module = importlib.util.module_from_spec(spec_hierarchical)
sys.modules['skillnote_recommendation.ml.unified_sem_estimator'] = unified_sem_module
spec_hierarchical.loader.exec_module(hierarchical_sem_module)

HierarchicalSEMEstimator = hierarchical_sem_module.HierarchicalSEMEstimator
DomainDefinition = hierarchical_sem_module.DomainDefinition


def generate_hierarchical_data(n_samples=500, n_skills_per_domain=10, seed=42):
    """
    階層構造を持つシミュレーションデータを生成

    構造:
    - 技術力（総合力量）
      - Python開発力（ドメイン）: 10スキル
      - Web開発力（ドメイン）: 10スキル
    - ビジネス力（総合力量）
      - 要件定義力（ドメイン）: 10スキル
      - 企画力（ドメイン）: 10スキル
    """
    np.random.seed(seed)

    # レベル3: 総合力量
    tech_ability = np.random.normal(0, 1, n_samples)
    business_ability = np.random.normal(0, 1, n_samples)

    # レベル2: ドメイン力量
    python_domain = 0.8 * tech_ability + np.random.normal(0, 0.3, n_samples)
    web_domain = 0.75 * tech_ability + np.random.normal(0, 0.35, n_samples)
    requirement_domain = 0.7 * business_ability + np.random.normal(0, 0.4, n_samples)
    planning_domain = 0.65 * business_ability + np.random.normal(0, 0.45, n_samples)

    # レベル1: 個別スキル
    data = {}

    # Python開発力のスキル
    for i in range(n_skills_per_domain):
        loading = np.random.uniform(0.6, 0.9)
        data[f'Python_skill_{i+1}'] = loading * python_domain + np.random.normal(0, 0.3, n_samples)

    # Web開発力のスキル
    for i in range(n_skills_per_domain):
        loading = np.random.uniform(0.6, 0.9)
        data[f'Web_skill_{i+1}'] = loading * web_domain + np.random.normal(0, 0.3, n_samples)

    # 要件定義力のスキル
    for i in range(n_skills_per_domain):
        loading = np.random.uniform(0.6, 0.9)
        data[f'Requirement_skill_{i+1}'] = loading * requirement_domain + np.random.normal(0, 0.3, n_samples)

    # 企画力のスキル
    for i in range(n_skills_per_domain):
        loading = np.random.uniform(0.6, 0.9)
        data[f'Planning_skill_{i+1}'] = loading * planning_domain + np.random.normal(0, 0.3, n_samples)

    return pd.DataFrame(data)


def main():
    print("=" * 80)
    print("HierarchicalSEMEstimator 動作確認テスト")
    print("=" * 80)

    # データ生成
    print("\n【1. データ生成】")
    n_samples = 500
    n_skills_per_domain = 10

    data = generate_hierarchical_data(n_samples, n_skills_per_domain)

    print(f"サンプルサイズ: n = {n_samples}")
    print(f"スキル数: {len(data.columns)} (各ドメイン{n_skills_per_domain}スキル)")
    print(f"\nスキル名の例:")
    print(f"  {list(data.columns[:5])}")

    # ドメイン定義
    print("\n【2. ドメイン定義】")

    domains = [
        # Level 1: スキル → ドメイン力量
        DomainDefinition(
            'Python開発力',
            [f'Python_skill_{i+1}' for i in range(n_skills_per_domain)],
            parent_domain='技術力',
            level=1
        ),
        DomainDefinition(
            'Web開発力',
            [f'Web_skill_{i+1}' for i in range(n_skills_per_domain)],
            parent_domain='技術力',
            level=1
        ),
        DomainDefinition(
            '要件定義力',
            [f'Requirement_skill_{i+1}' for i in range(n_skills_per_domain)],
            parent_domain='ビジネス力',
            level=1
        ),
        DomainDefinition(
            '企画力',
            [f'Planning_skill_{i+1}' for i in range(n_skills_per_domain)],
            parent_domain='ビジネス力',
            level=1
        ),

        # Level 2: ドメイン力量 → 総合力量
        DomainDefinition(
            '技術力',
            ['Python開発力', 'Web開発力'],
            level=2
        ),
        DomainDefinition(
            'ビジネス力',
            ['要件定義力', '企画力'],
            level=2
        ),
    ]

    print(f"Level 1ドメイン（スキル→ドメイン）:")
    for d in domains:
        if d.level == 1:
            print(f"  - {d.domain_name}: {len(d.skills)}スキル → 親: {d.parent_domain}")

    print(f"\nLevel 2ドメイン（ドメイン→総合）:")
    for d in domains:
        if d.level == 2:
            print(f"  - {d.domain_name}: {d.skills}")

    # モデル推定（逐次処理）
    print("\n【3. 階層的SEM推定（逐次処理）】")
    hsem = HierarchicalSEMEstimator(domains, method='ML')

    start_time = time.time()
    result = hsem.fit(data, n_jobs=1)
    elapsed_sequential = time.time() - start_time

    print(f"\n実行時間: {elapsed_sequential:.2f}秒")
    print(f"推定されたドメインモデル数: {result.n_domains}")

    # 適合度指標
    print("\n【4. 適合度指標】")

    print("\n■ ドメインモデル別:")
    for domain_name, fit in result.domain_fit_indices.items():
        status = "✅" if fit.is_good_fit() else "⚠️ "
        print(f"  {status} {domain_name}:")
        print(f"      RMSEA={fit.rmsea:.3f}, CFI={fit.cfi:.3f}, TLI={fit.tli:.3f}")

    if result.integration_fit_indices:
        print("\n■ 統合モデル:")
        fit = result.integration_fit_indices
        status = "✅" if fit.is_good_fit() else "⚠️ "
        print(f"  {status} RMSEA={fit.rmsea:.3f}, CFI={fit.cfi:.3f}, TLI={fit.tli:.3f}")

    print("\n■ 全体適合度:")
    overall = result.overall_fit
    print(f"  RMSEA: {overall['rmsea']:.3f}")
    print(f"  CFI: {overall['cfi']:.3f}")
    print(f"  TLI: {overall['tli']:.3f}")

    # ドメインスコア
    print("\n【5. ドメインスコア】")
    print(f"\nドメインスコア（最初の5人）:")
    print(result.domain_scores.head())

    print(f"\nドメインスコアの統計量:")
    print(result.domain_scores.describe().T)

    # スキル→ドメインのローディング
    print("\n【6. スキル→ドメインのファクターローディング】")
    loadings = hsem.get_skill_to_domain_loadings()

    print(f"\n各ドメインの平均ローディング:")
    avg_loadings = loadings.groupby('domain')['loading'].mean().sort_values(ascending=False)
    for domain, avg_loading in avg_loadings.items():
        print(f"  {domain}: {avg_loading:.3f}")

    print(f"\n高いローディングのスキル（上位10件）:")
    top_loadings = loadings.nlargest(10, 'loading')
    for _, row in top_loadings.iterrows():
        print(f"  {row['skill']} → {row['domain']}: {row['loading']:.3f}")

    # 並列処理のテスト
    print("\n【7. 並列処理のテスト】")

    hsem_parallel = HierarchicalSEMEstimator(domains, method='ML')

    start_time = time.time()
    result_parallel = hsem_parallel.fit(data, n_jobs=4, use_multiprocessing=False)
    elapsed_parallel = time.time() - start_time

    print(f"\n逐次処理: {elapsed_sequential:.2f}秒")
    print(f"並列処理(4スレッド): {elapsed_parallel:.2f}秒")
    print(f"高速化率: {elapsed_sequential / elapsed_parallel:.2f}x")

    # 全レベルのスコア予測
    print("\n【8. 全レベルのスコア予測】")

    all_scores = hsem.predict_all_scores(data)

    print(f"\nドメインスコア:")
    print(all_scores['domain_scores'].head())

    if 'total_scores' in all_scores:
        print(f"\n総合力量スコア:")
        print(all_scores['total_scores'].head())

    print("\n" + "=" * 80)
    print("テスト完了！")
    print("=" * 80)

    # 結果サマリー
    print("\n【結果サマリー】")
    print(f"✅ {result.n_domains}個のドメインモデルを推定")
    print(f"✅ {result.n_skills}個のスキルを分析")
    print(f"✅ 実行時間: {result.elapsed_time:.2f}秒")
    print(f"✅ 全体適合度: {'良好' if result.overall_fit['cfi'] > 0.90 else '要改善'}")

    print("\n【スケーラビリティ見積もり】")
    # 1000スキルの場合の見積もり
    skills_ratio = 1000 / result.n_skills
    estimated_time_1000 = result.elapsed_time * skills_ratio * 0.8  # 並列処理で20%削減
    print(f"  スキル1000個の場合の推定実行時間: {estimated_time_1000:.1f}秒")
    print(f"  （現在: {result.n_skills}スキル, {result.elapsed_time:.2f}秒）")

    return hsem, result


if __name__ == '__main__':
    hsem, result = main()
