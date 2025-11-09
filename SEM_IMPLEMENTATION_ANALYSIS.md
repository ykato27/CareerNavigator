# CareerNavigator SEM (構造方程式モデリング) 実装分析レポート

## 1. SEM関連ファイル一覧

### コアモデル実装
1. **skill_domain_sem_model.py** (970行)
   - スキル領域潜在変数SEMモデル（正しいSEM実装版）
   - 測定モデル + 構造モデル + 統計的推定

2. **skill_dependency_sem_model.py** (470行)
   - スキル依存関係SEM（観測変数ベース）
   - スキル間の直接的な因果関係分析

### レコメンダー実装
3. **sem_only_recommender.py** (471行)
   - SEM専用推薦エンジン
   - NMFなしで完全にSEMベース

4. **ml_sem_recommender.py** (435行)
   - ML推薦エンジンへのSEM統合
   - ハイブリッド推薦システム

### UI/ページ
5. **3_SEM_Analysis.py** (909行)
   - インタラクティブなSEM分析ページ
   - 領域別プロファイル、ネットワーク可視化

### テスト
6. **test_skill_domain_sem_model.py** (288行)
   - SkillDomainSEMModelの包括的テスト

---

## 2. 目的関数（Objective Function）の実装

### 2.1 パス係数推定の目的関数

**スキル依存関係SEM** (`skill_dependency_sem_model.py`)
```
目的: スキル間の因果関係を単回帰で推定
推定式: Y = a + b*X + e

単回帰係数（パス係数）:
b = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)

実装位置: _estimate_path_coefficient_fast() メソッド (行171-248)
```

**コア計算**:
```python
numerator = np.dot(x_diff, y_diff)      # 分子: 共分散
denominator = np.dot(x_diff, x_diff)    # 分母: x の分散
coefficient = numerator / denominator   # パス係数
```

### 2.2 構造モデルの目的関数

**スキル領域潜在変数SEM** (`skill_domain_sem_model.py`)
```
目的: 潜在変数間の因果効果を相関係数で推定

構造モデルのパス係数:
coefficient = Pearson相関係数 (r)

実装位置: _calculate_path_coefficient() メソッド (行465-537)
```

**コア計算**:
```python
# ピアソン相関係数
coefficient = np.corrcoef(from_std, to_std)[0, 1]

# t値計算
t_value = coefficient * √(n-2) / √(1 - coefficient²)

# p値計算（両側検定）
p_value = 2 * (1 - t.cdf(|t_value|, n-2))
```

### 2.3 測定モデルの目的関数

```
目的: スキルレベル → 潜在変数の関係を推定

ファクターローディング (λ):
λ = std(skill_levels)  # スキルレベルの標準偏差

測定誤差分散:
δ = 1 - λ²

実装位置: _estimate_measurement_model() メソッド (行296-373)
```

### 2.4 推定方法の体系

| 推定対象 | 推定方法 | 統計量 |
|---------|--------|------|
| パス係数（スキル依存）| OLS (単回帰) | t値、p値、信頼区間 |
| パス係数（潜在変数）| ピアソン相関 | t値、p値、信頼区間 |
| ファクターローディング | 標準偏差推定 | 因子負荷量 |
| 測定誤差分散 | 1 - λ² | 誤差分散 |

---

## 3. スキル間の関係性（共分散・パス係数）の扱い

### 3.1 スキル依存関係SEM (観測変数ベース)

**データ構造**:
```
メンバー × スキル マトリックス
    ↓ (pivot_table)
スキルレベル行列
    ↓
パス係数計算
```

**実装フロー** (行96-152):
```python
# 1. スキルレベルマトリックスをピボット
skill_levels = member_competence_df.pivot_table(
    index='メンバーコード',
    columns='力量コード',
    values='正規化レベル',
    fill_value=0
)

# 2. すべてのスキルペアを分析
for from_skill in skills:
    for to_skill in skills:
        # 3. パス係数を推定
        from_levels = skill_levels[from_skill].values
        to_levels = skill_levels[to_skill].values
        
        # 4. 有効なデータペアのみを使用
        valid_mask = (from_levels > 0) & (to_levels > 0)
        
        # 5. パス係数を計算
        path_coeff = _estimate_path_coefficient_fast(...)
        
        # 6. 統計的有意性でフィルタ (p < 0.05)
        if path_coeff.is_significant:
            skill_paths.append(path_coeff)
```

### 3.2 スキル領域潜在変数SEM (潜在変数ベース)

**モデル構造**:
```
観測スキル → 潜在変数（初級/中級/上級） → 潜在変数
             ↓
           測定モデル
             ↓
        パス係数推定
```

**実装フロー** (行214-254, 429-463):

1. **測定モデル推定**:
   - 観測スキル → 潜在変数の関係
   - ファクターローディング計算

2. **潜在スコア推定** (行375-427):
   ```python
   # メンバーの潜在変数スコア = 加重平均
   latent_score = Σ(スキルレベル × ローディング) / Σ(ローディング)
   ```

3. **構造モデル推定** (行429-463):
   ```python
   # 初級 → 中級 → 上級 の段階的パス
   from_scores = [メンバーの初級スコア]
   to_scores = [メンバーの中級スコア]
   
   path_coeff = ピアソン相関(from_scores, to_scores)
   ```

### 3.3 共分散の計算

**相関行列計算** (行153-169):
```python
# スキル間の相関行列
correlation_matrix = skill_levels.corr(method='pearson')
```

**共分散の活用**:
- スキル依存関係SEMではパス係数の計算に使用
- 潜在変数間のパス係数計算でも使用
- ただし、共分散行列は明示的には保存されない

---

## 4. 推定方法の詳細

### 4.1 OLS推定（スキル依存関係）

**実装** (行188-231 in skill_dependency_sem_model.py):

```python
# 1. データ準備
from_levels = スキルXのレベル
to_levels = スキルYのレベル

# 2. 平均を計算
mean_x = np.mean(from_levels)
mean_y = np.mean(to_levels)

# 3. 差分を計算
x_diff = from_levels - mean_x
y_diff = to_levels - mean_y

# 4. パス係数を計算（最小二乗推定）
numerator = np.dot(x_diff, y_diff)      # Σ(xi - x̄)(yi - ȳ)
denominator = np.dot(x_diff, x_diff)    # Σ(xi - x̄)²
coefficient = numerator / denominator   # β = covariance / variance_X

# 5. 残差を計算
residuals = y_diff - coefficient * x_diff

# 6. 標準誤差を計算
ss_residual = np.dot(residuals, residuals)
mse = ss_residual / (n - 2)
se_coefficient = np.sqrt(mse / denominator)

# 7. t値を計算
t_value = coefficient / se_coefficient

# 8. p値を計算
p_value = 2 * (1 - t.cdf(abs(t_value), n - 2))

# 9. 信頼区間を計算
t_critical = t.ppf((1 + confidence_level) / 2, n - 2)
ci_lower = coefficient - t_critical * se_coefficient
ci_upper = coefficient + t_critical * se_coefficient
```

**統計量の解釈**:
- coefficient: パス係数（-1～1）
- t_value: 統計量（値が大きいほど有意）
- p_value: 有意性（p < 0.05で有意）
- ci_lower, ci_upper: 95%信頼区間

### 4.2 相関係数ベース推定（潜在変数間）

**実装** (行491-523 in skill_domain_sem_model.py):

```python
# 1. 標準化スコア
from_std = (from_array - mean) / std
to_std = (to_array - mean) / std

# 2. ピアソン相関係数
coefficient = np.corrcoef(from_std, to_std)[0, 1]

# 3. t値計算（相関係数用）
t_value = r * √(n-2) / √(1-r²)

# 4. p値計算
p_value = 2 * (1 - t.cdf(abs(t_value), n - 2))

# 5. フィッシャーのz変換で信頼区間
z = 0.5 * ln((1 + r) / (1 - r))
se_z = 1 / √(n - 3)
z_critical = norm.ppf((1 + confidence_level) / 2)

ci_lower = tanh(z - z_critical * se_z)
ci_upper = tanh(z + z_critical * se_z)
```

### 4.3 ファクターローディング推定（測定モデル）

**実装** (行332-373):

```python
# スキルレベルの標準偏差をローディングとして使用
loading = np.std(skill_levels)

# 正規化（0.3-0.95の範囲）
loading = min(max(loading, 0.3), 0.95)

# 測定誤差分散
measurement_error = 1.0 - loading²

# Cronbach's alphaの簡易推定
if len(factor_loadings) > 1:
    item_reliability = (
        (k * mean_loading) /
        (1 + (k - 1) * 0.5)
    )
```

---

## 5. 適合度指標（Fit Indices）の実装

### 5.1 実装位置

`skill_domain_sem_model.py` の `get_model_fit_indices()` メソッド (行680-743)

### 5.2 実装された指標

| 指標 | 計算方法 | 望ましい値 | 実装行 |
|-----|--------|---------|------|
| **GFI** (Goodness of Fit Index) | `(sig_ratio × 0.5) + (avg_loading × 0.5)` | ≥ 0.9 | 726 |
| **NFI** (Normed Fit Index) | `平均\|パス係数\|(有意なパスのみ)` | ≥ 0.9 | 731 |
| **RMSEA** | ❌ 未実装 | < 0.08 | - |
| **CFI** | ❌ 未実装 | ≥ 0.95 | - |
| **TLI** | ❌ 未実装 | ≥ 0.95 | - |
| **R² (説明分散)** | `mean(loading²)` | 1に近い | 720 |

### 5.3 実装コード

```python
def get_model_fit_indices(self, domain_name: str) -> Dict[str, float]:
    """モデル適合度指標を計算"""
    
    domain_struct = self.domain_structures.get(domain_name)
    
    # パス係数の統計
    path_coeffs = [p.coefficient for p in domain_struct.path_coefficients]
    significant_paths = sum(
        1 for p in domain_struct.path_coefficients if p.is_significant
    )
    
    # 平均因子負荷量
    loadings = [
        loading
        for f in domain_struct.latent_factors
        for loading in f.factor_loadings.values()
    ]
    avg_loading = np.mean(loadings) if loadings else 0.0
    
    # 効果サイズ（Cohen's d）
    effect_sizes = [abs(p.coefficient) for p in domain_struct.path_coefficients]
    avg_effect_size = np.mean(effect_sizes) if effect_sizes else 0.0
    
    # 説明分散（R²）
    variance_explained = np.mean([l**2 for l in loadings]) if loadings else 0.0
    
    # GFI（簡易推定）
    sig_ratio = significant_paths / len(domain_struct.path_coefficients)
    gfi = (sig_ratio * 0.5) + (avg_loading * 0.5)
    
    # NFI（簡易推定）
    nfi = (mean|coefficient|(有意パス) if 有意パスあり else 0.0)
    
    return {
        "avg_path_coefficient": np.mean(path_coeffs) if path_coeffs else 0.0,
        "significant_paths": significant_paths,
        "total_paths": len(domain_struct.path_coefficients),
        "avg_loading": avg_loading,
        "avg_effect_size": avg_effect_size,
        "variance_explained": variance_explained,
        "gfi": min(gfi, 1.0),  # 0-1に制限
        "nfi": min(nfi, 1.0),  # 0-1に制限
    }
```

### 5.4 実装の制限事項

- **GFIとNFI**: 簡易推定のみ（標準的な計算式ではない）
- **RMSEA、CFI、TLI**: 未実装
- **モデル比較指標（AIC、BIC）**: 未実装
- **効果サイズ**: Cohen's dの簡易推定のみ

---

## 6. 力量（スキル）同士の関係性を目的関数でどう分析しているか

### 6.1 スキル依存関係SEM（直接的な因果分析）

**アプローチ**:
```
スキル X → スキル Y への因果効果を単回帰で直接推定
```

**目的関数**:
```
目的: スキルペア（X,Y）について、Y = a + b*X + e を推定し、
      パス係数 b を求めることで X → Y の因果効果を定量化

実装:
- すべてのスキルペアに対して単回帰を実行
- b (パス係数) を推定
- t検定で統計的有意性を判定 (p < 0.05)
- 有意なペアのみをスキル依存関係として保持
```

**具体例** (skill_dependency_sem_model.py 行96-152):

```
スキルペア: Python基礎 → Webアプリ開発

Python基礎のレベル: [3, 4, 2, 5, ...]
Webアプリのレベル: [4, 5, 3, 5, ...]

↓ OLS推定

パス係数 b = 0.68
t値 = 3.42
p値 = 0.012 < 0.05 ✓ (有意)

解釈: Python基礎のレベルが1上がると、
     Webアプリ開発のレベルが0.68上がる傾向がある
```

### 6.2 スキル領域潜在変数SEM（段階的な習得構造分析）

**アプローチ**:
```
潜在変数 L₁(初級) → L₂(中級) → L₃(上級) の段階的な因果効果を分析
```

**目的関数**:
```
目的: 各メンバーについて潜在変数スコアを推定し、
      レベル間のパス係数（相関係数）を計算

実装:
1. 測定モデル: スキル → 潜在変数（ファクターローディング）
   λ = std(スキルレベル)
   
2. 潜在スコア推定: 
   L = Σ(スキルレベル × ローディング) / Σ(ローディング)
   
3. 構造モデル:
   パス係数 = Pearson相関(L₁スコア, L₂スコア)
   
4. 統計的検定:
   t値 = r * √(n-2) / √(1-r²)
   p値 = 2 * P(T > |t値|, n-2)
```

**具体例** (skill_domain_sem_model.py 行375-537):

```
プログラミング領域の構造分析

メンバー: M001, M002, M003, ...

初級スコア:      [0.5, 0.4, 0.6, ...]
中級スコア:      [0.6, 0.5, 0.7, ...]
上級スコア:      [0.8, 0.6, 0.9, ...]

↓ 相関係数計算

初級 → 中級: r = 0.82, t = 4.12, p = 0.001 ✓ (有意)
中級 → 上級: r = 0.75, t = 3.45, p = 0.005 ✓ (有意)

解釈: 初級スキルが高いほど中級スキルを習得しやすい傾向がある
```

### 6.3 SEMスコアの計算（推薦スコアへの統合）

**アプローチ**:
```
メンバーの現在のレベルから、次のレベルへの習得確率を推定
```

**実装** (skill_domain_sem_model.py 行539-584):

```python
def calculate_sem_score(self, member_code, skill_code):
    """SEMスコア = 現在スコア × パス係数"""
    
    # 1. スキルの所属領域を特定
    domain = _find_skill_domain(skill_code)
    
    # 2. メンバーの現在のレベルを推定
    current_level = _estimate_current_level(member_code, domain)
    
    # 3. 次のレベルへのパス係数を取得
    path_coef = domain_struct.path_coefficients[current_level]
    
    # 4. SEMスコア = 現在スコア × パス係数
    if path_coef.is_significant:
        sem_score = current_score * path_coef.coefficient
    else:
        sem_score = current_score * 0.6  # デフォルト値
    
    return min(1.0, max(0.0, sem_score))
```

**意義**:
- パス係数が大きい → 習得確率が高い
- パス係数が小さい → 習得確率が低い
- 統計的に有意でない → デフォルト値(0.6)を使用

---

## 7. 実装の特徴と制限事項

### 7.1 実装の特徴

✅ **利点**:
1. **実装の単純性**: OLS推定、相関係数など基本的な統計
2. **計算効率**: 簡易推定により高速な計算が可能
3. **解釈可能性**: パス係数が直感的に理解しやすい
4. **スケーラビリティ**: 大規模データでも処理可能
5. **統計的根拠**: t検定、p値による有意性判定

### 7.2 制限事項

❌ **制限**:
1. **標準的SEM手法の未実装**:
   - ML (Maximum Likelihood) 推定なし
   - GLS (Generalized Least Squares) なし
   - WLS (Weighted Least Squares) なし

2. **適合度指標の簡易推定**:
   - GFI, NFIは標準的な計算式ではない
   - RMSEA, CFI, TLI未実装
   - モデル比較指標なし

3. **測定モデルの簡易実装**:
   - ファクターローディング = 標準偏差（理論的根拠が弱い）
   - CFA (Confirmatory Factor Analysis) なし

4. **構造モデルの簡略化**:
   - 相関係数ベース（因果効果の厳密な推定ではない）
   - 多変量パス分析なし
   - 媒介効果分析が限定的

5. **データの仮定**:
   - 正規性の検定なし
   - 共線性の診断なし
   - 外れ値検出なし

---

## 8. スキル間の関係性分析の具体例

### 例：Webアプリ開発の習得因果構造

```
SkillDependencySEMModel による分析:

スキル依存関係:
    Python基礎 (0.72) → Webアプリ開発
    JavaScript基礎 (0.65) → Webアプリ開発
    データベース基礎 (0.58) → Webアプリ開発
    
パス係数: (0-1)
t値: 統計量
p値: 有意性

SkillDomainSEMModel による分析:

潜在変数構造:
    初級（Python, JS基礎）
        ↓ (r=0.82, p=0.001)
    中級（Webアプリ基礎）
        ↓ (r=0.75, p=0.005)
    上級（Webアプリ応用）

メンバーM001の推薦スコア:
    SEMスコア = 中級スコア(0.65) × パス係数(0.75) = 0.49
```

---

## 9. まとめ

### SEM実装の体系

| 層 | 手法 | 推定対象 | 実装ファイル |
|---|---|----|---------|
| **観測レベル** | OLS単回帰 | スキル → スキル | skill_dependency_sem_model.py |
| **潜在レベル（測定）** | 標準偏差推定 | スキル → 潜在変数 | skill_domain_sem_model.py |
| **潜在レベル（構造）** | ピアソン相関 | 潜在変数 → 潜在変数 | skill_domain_sem_model.py |
| **推薦統合** | スコア加重 | 習得確率推定 | sem_only_recommender.py |

### 実装戦略

- **簡易・実用的**: 標準SEM手法ではなく、実装が容易な方法を採用
- **速度優先**: 複雑な最適化計算を避け、解析的な計算を実施
- **解釈性重視**: パス係数が直感的に理解できる設計
- **統計的厳密性**: t検定、p値による有意性判定は実施

### 推薦への活用

SEMスコア = 現在の習得度 × パス係数（次レベルへの因果効果）

このアプローチにより、単なる統計的関連性ではなく、
実際の習得構造に基づいた推薦が可能になっている。

