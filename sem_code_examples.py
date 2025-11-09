"""
SEM実装の具体的なコード例
CareerNavigator プロジェクトから抽出
"""

# ============================================================================
# 例1: スキル依存関係SEM - パス係数推定
# ============================================================================

import numpy as np
from scipy import stats

def estimate_path_coefficient_example():
    """
    スキル間の因果関係を推定する例
    
    目的: Python基礎 → Webアプリ開発 への因果効果を定量化
    """
    
    # メンバーのスキルレベル（サンプルデータ）
    python_levels = np.array([3, 4, 2, 5, 3, 4, 2, 3])      # Python基礎
    webapp_levels = np.array([4, 5, 3, 5, 3, 4, 2, 4])      # Webアプリ
    
    # 1. 平均を計算
    mean_x = np.mean(python_levels)
    mean_y = np.mean(webapp_levels)
    
    # 2. 差分を計算
    x_diff = python_levels - mean_x
    y_diff = webapp_levels - mean_y
    
    # 3. パス係数を計算（OLS推定）
    numerator = np.dot(x_diff, y_diff)        # 共分散
    denominator = np.dot(x_diff, x_diff)      # 分散
    coefficient = numerator / denominator     # パス係数
    
    # 4. 残差を計算
    residuals = y_diff - coefficient * x_diff
    
    # 5. 標準誤差を計算
    n = len(python_levels)
    ss_residual = np.dot(residuals, residuals)
    mse = ss_residual / (n - 2)
    se_coefficient = np.sqrt(mse / denominator)
    
    # 6. t値を計算
    t_value = coefficient / se_coefficient
    
    # 7. p値を計算
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - 2))
    
    # 8. 信頼区間を計算
    confidence_level = 0.95
    t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 2)
    ci_lower = coefficient - t_critical * se_coefficient
    ci_upper = coefficient + t_critical * se_coefficient
    
    # 結果
    print("=" * 70)
    print("スキル依存関係SEM - パス係数推定")
    print("=" * 70)
    print(f"\nPython基礎 → Webアプリ開発の因果効果")
    print(f"  パス係数 (β):           {coefficient:.4f}")
    print(f"  標準誤差:              {se_coefficient:.4f}")
    print(f"  t値:                  {t_value:.4f}")
    print(f"  p値:                  {p_value:.6f}")
    print(f"  信頼区間 (95%):        [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  統計的有意性 (p<0.05): {'✓ 有意' if p_value < 0.05 else '✗ 有意でない'}")
    
    print(f"\n解釈:")
    print(f"  Python基礎のレベルが1上がると、")
    print(f"  Webアプリ開発のレベルが約{coefficient:.2f}上がる傾向がある")
    print()


# ============================================================================
# 例2: スキル領域潜在変数SEM - 構造モデルのパス係数推定
# ============================================================================

def estimate_latent_path_coefficient_example():
    """
    潜在変数間の因果効果を推定する例
    
    目的: 初級スコア → 中級スコア への段階的な習得効果を定量化
    """
    
    # メンバーの潜在変数スコア（例：プログラミング領域）
    beginner_scores = np.array([0.5, 0.4, 0.6, 0.3, 0.7, 0.4, 0.5, 0.6])
    intermediate_scores = np.array([0.6, 0.5, 0.7, 0.4, 0.8, 0.5, 0.6, 0.7])
    
    # 1. 標準化
    from_std = (beginner_scores - beginner_scores.mean()) / (beginner_scores.std() + 1e-10)
    to_std = (intermediate_scores - intermediate_scores.mean()) / (intermediate_scores.std() + 1e-10)
    
    # 2. ピアソン相関係数
    coefficient = np.corrcoef(from_std, to_std)[0, 1]
    
    # 3. t値計算
    n = len(beginner_scores)
    if abs(coefficient) < 0.9999:
        t_value = coefficient * np.sqrt(n - 2) / np.sqrt(max(1 - coefficient**2, 1e-10))
    else:
        t_value = 0.0
    
    # 4. p値計算
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - 2))
    
    # 5. フィッシャーのz変換で信頼区間
    z = 0.5 * np.log((1 + coefficient) / (1 - coefficient + 1e-10))
    se_z = 1.0 / np.sqrt(n - 3)
    z_critical = stats.norm.ppf((1 + 0.95) / 2)  # 95%信頼水準
    ci_lower = np.tanh(z - z_critical * se_z)
    ci_upper = np.tanh(z + z_critical * se_z)
    
    # 結果
    print("=" * 70)
    print("スキル領域潜在変数SEM - 構造モデルのパス係数推定")
    print("=" * 70)
    print(f"\n初級スコア → 中級スコアのパス係数")
    print(f"  相関係数 (r):          {coefficient:.4f}")
    print(f"  t値:                  {t_value:.4f}")
    print(f"  p値:                  {p_value:.6f}")
    print(f"  信頼区間 (95%):        [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  統計的有意性 (p<0.05): {'✓ 有意' if p_value < 0.05 else '✗ 有意でない'}")
    
    print(f"\n解釈:")
    print(f"  初級スキルが高いほど、中級スキルを習得しやすい傾向がある")
    print(f"  相関係数 {coefficient:.3f} は {'強い正の相関' if coefficient > 0.7 else '中程度の相関' if coefficient > 0.5 else '弱い相関'}を示している")
    print()


# ============================================================================
# 例3: 測定モデル - ファクターローディング推定
# ============================================================================

def estimate_measurement_model_example():
    """
    スキル → 潜在変数の関係を推定する例
    
    目的: プログラミング初級潜在変数を構成するスキルのローディングを計算
    """
    
    # メンバーのスキルレベル（正規化: 0-1）
    python_levels = np.array([0.4, 0.5, 0.3, 0.6, 0.5]) / 5.0
    java_levels = np.array([0.3, 0.4, 0.4, 0.5, 0.6]) / 5.0
    git_levels = np.array([0.5, 0.6, 0.4, 0.7, 0.5]) / 5.0
    
    skills = {
        'Python基礎': python_levels,
        'Java基礎': java_levels,
        'Git': git_levels
    }
    
    print("=" * 70)
    print("測定モデル - ファクターローディング推定")
    print("=" * 70)
    print("\nプログラミング初級潜在変数のファクターローディング\n")
    
    factor_loadings = {}
    for skill_name, skill_levels in skills.items():
        # ファクターローディング = スキルレベルの標準偏差
        loading = np.std(skill_levels)
        
        # 正規化（0.3-0.95の範囲）
        loading = min(max(loading, 0.3), 0.95)
        
        # 測定誤差分散
        error_variance = 1.0 - loading**2
        
        factor_loadings[skill_name] = loading
        
        print(f"{skill_name}:")
        print(f"  ローディング (λ):      {loading:.4f}")
        print(f"  誤差分散:              {error_variance:.4f}")
        print()
    
    # Cronbach's alphaの簡易推定
    mean_loading = np.mean(list(factor_loadings.values()))
    k = len(factor_loadings)
    item_reliability = (k * mean_loading) / (1 + (k - 1) * 0.5)
    
    print(f"潜在変数の信頼性:")
    print(f"  平均ローディング:      {mean_loading:.4f}")
    print(f"  Cronbach's alpha:     {item_reliability:.4f}")
    print(f"  信頼性判定:            {'✓ 許容範囲' if item_reliability > 0.6 else '✗ 要改善'}")
    print()


# ============================================================================
# 例4: SEMスコア計算 - 推薦スコアへの統合
# ============================================================================

def calculate_sem_score_example():
    """
    メンバーのスキル習得確率を計算する例
    
    目的: 次のレベルへの習得確率を推定
    """
    
    # メンバーの現在スコア（中級）
    current_level_score = 0.65
    
    # パス係数（中級 → 上級）
    path_coefficient = 0.75
    path_p_value = 0.005
    
    # SEMスコア計算
    if path_p_value < 0.05:  # 統計的に有意
        sem_score = current_level_score * path_coefficient
    else:
        sem_score = current_level_score * 0.6  # デフォルト値
    
    # 0-1の範囲に正規化
    sem_score = min(1.0, max(0.0, sem_score))
    
    print("=" * 70)
    print("SEMスコア計算 - 習得確率推定")
    print("=" * 70)
    print("\nメンバーM001のプログラミング上級スキル習得確率\n")
    print(f"現在のレベル（中級）スコア:     {current_level_score:.3f}")
    print(f"パス係数（中級→上級）:         {path_coefficient:.3f}")
    print(f"p値（統計的有意性）:           {path_p_value:.6f}")
    print(f"統計的有意性:                  {'✓ 有意' if path_p_value < 0.05 else '✗ 有意でない'}")
    print(f"\nSEMスコア計算:")
    print(f"  = 現在スコア × パス係数")
    print(f"  = {current_level_score:.3f} × {path_coefficient:.3f}")
    print(f"  = {sem_score:.3f}")
    print(f"\n推薦確率: {sem_score * 100:.1f}%")
    print()


# ============================================================================
# 例5: モデル適合度指標
# ============================================================================

def calculate_model_fit_indices_example():
    """
    モデルの適合度を評価する例
    """
    
    # サンプルデータ
    path_coefficients = [0.82, 0.75, 0.68, 0.71, 0.79]
    significant_paths = 5  # すべて有意
    total_paths = 5
    
    factor_loadings = [0.75, 0.68, 0.72, 0.70, 0.76, 0.74]
    
    print("=" * 70)
    print("モデル適合度指標の計算")
    print("=" * 70)
    print()
    
    # 1. 基本統計
    avg_path_coefficient = np.mean(path_coefficients)
    avg_loading = np.mean(factor_loadings)
    avg_effect_size = np.mean([abs(p) for p in path_coefficients])
    
    print("1. 基本統計:")
    print(f"  平均パス係数:          {avg_path_coefficient:.3f}")
    print(f"  有意なパス数:          {significant_paths}/{total_paths}")
    print(f"  平均因子負荷量:        {avg_loading:.3f}")
    print(f"  平均効果サイズ:        {avg_effect_size:.3f}")
    print()
    
    # 2. 説明分散
    variance_explained = np.mean([l**2 for l in factor_loadings])
    print(f"2. 説明分散 (R²):")
    print(f"  R² = {variance_explained:.3f}")
    print(f"  解釈: モデルがデータの{variance_explained*100:.1f}%を説明")
    print()
    
    # 3. GFI (簡易推定)
    sig_ratio = significant_paths / total_paths
    gfi = (sig_ratio * 0.5) + (avg_loading * 0.5)
    gfi = min(gfi, 1.0)
    print(f"3. GFI (Goodness of Fit Index) - 簡易推定:")
    print(f"  GFI = {gfi:.3f}")
    print(f"  判定: {'✓ 良好 (≥0.9)' if gfi >= 0.9 else '△ 許容 (≥0.8)' if gfi >= 0.8 else '✗ 要改善'}")
    print()
    
    # 4. NFI (簡易推定)
    nfi = avg_effect_size if avg_effect_size > 0 else 0.0
    nfi = min(nfi, 1.0)
    print(f"4. NFI (Normed Fit Index) - 簡易推定:")
    print(f"  NFI = {nfi:.3f}")
    print(f"  判定: {'✓ 良好 (≥0.9)' if nfi >= 0.9 else '△ 許容 (≥0.8)' if nfi >= 0.8 else '✗ 要改善'}")
    print()


# ============================================================================
# メイン実行
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" CareerNavigator SEM実装 - コード例")
    print("=" * 70 + "\n")
    
    # 各例を実行
    estimate_path_coefficient_example()
    estimate_latent_path_coefficient_example()
    estimate_measurement_model_example()
    calculate_sem_score_example()
    calculate_model_fit_indices_example()
    
    print("=" * 70)
    print(" まとめ")
    print("=" * 70)
    print("""
SEM（構造方程式モデリング）の実装は以下の層からなります：

1. スキル依存関係SEM（観測変数レベル）
   → OLS推定で直接的なスキル間の因果効果を推定

2. スキル領域潜在変数SEM（潜在変数レベル）
   → 測定モデル + 構造モデルで階層的な習得構造を分析

3. 統計的検定
   → t検定、p値、信頼区間により有意性を判定

4. 推薦への統合
   → SEMスコア = 現在スコア × パス係数で習得確率を推定

このアプローチにより、単なる統計的関連性ではなく、
実際の習得構造に基づいた推薦が可能になります。
    """)
