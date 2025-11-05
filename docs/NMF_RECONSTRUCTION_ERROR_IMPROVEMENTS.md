# NMF再構成誤差改善提案

## 📊 現状分析

### 現在の設定

**ml_recommender.py:63**
```python
mf_model = MatrixFactorizationModel(n_components=20, random_state=42)
```

**matrix_factorization.py:33-39**
```python
default_params = {'init': 'nndsvda', 'max_iter': 500}
self.model = NMF(
    n_components=n_components,
    random_state=random_state,
    **final_params
)
```

### 問題点

1. **設定の不統一**: Config.MF_PARAMSが定義されているが使用されていない
2. **パラメータ最適化の欠如**: ハードコードされた値で固定
3. **正則化なし**: alphaパラメータ（L1/L2正則化）が未設定
4. **収束判定の甘さ**: tolパラメータがデフォルト値(1e-4)のまま
5. **データ特性の未考慮**: スパース性や欠損値の扱いが最適化されていない

---

## 🎯 改善提案

### 提案1: Configベースのパラメータ管理（即効性★★★★★）

**効果**: コードの一貫性向上、パラメータ調整の容易化

**実装**:

#### 1.1 Config.pyの拡張

```python
# Matrix Factorization パラメータ
MF_PARAMS = {
    # 基本パラメータ
    'n_components': 20,  # 潜在因子の数（10-30推奨）
    'max_iter': 1000,  # 最大イテレーション数（500-2000推奨）
    'random_state': 42,  # 再現性のための乱数シード

    # 収束パラメータ
    'tol': 1e-5,  # 収束判定の閾値（1e-4 → 1e-5で精度向上）

    # 初期化戦略
    'init': 'nndsvda',  # 'nndsvda', 'nndsvd', 'random'

    # 正則化パラメータ（重要！）
    'alpha_W': 0.01,  # メンバー因子行列のL1正則化（0.0-0.1推奨）
    'alpha_H': 0.01,  # 力量因子行列のL1正則化（0.0-0.1推奨）
    'l1_ratio': 0.5,  # L1正則化の割合（0.0=L2のみ, 1.0=L1のみ）

    # ソルバー
    'solver': 'cd',  # 'cd' (coordinate descent) or 'mu' (multiplicative update)

    # その他
    'beta_loss': 'frobenius',  # 'frobenius', 'kullback-leibler', 'itakura-saito'
}

# パラメータチューニング用の探索範囲
MF_PARAMS_SEARCH_SPACE = {
    'n_components': [10, 15, 20, 25, 30],
    'alpha_W': [0.0, 0.001, 0.01, 0.05, 0.1],
    'alpha_H': [0.0, 0.001, 0.01, 0.05, 0.1],
    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
}
```

#### 1.2 ml_recommender.pyの修正

```python
from skillnote_recommendation.core.config import Config

@classmethod
def build(cls, member_competence: pd.DataFrame,
          competence_master: pd.DataFrame,
          member_master: pd.DataFrame):
    # ...

    # Configから設定を読み込み
    mf_params = Config.MF_PARAMS.copy()
    n_components = mf_params.pop('n_components')
    random_state = mf_params.pop('random_state')

    # NMFモデルを学習
    mf_model = MatrixFactorizationModel(
        n_components=n_components,
        random_state=random_state,
        **mf_params  # 正則化パラメータなどを渡す
    )
    mf_model.fit(skill_matrix)

    # ...
```

---

### 提案2: 正則化の導入（効果★★★★★）

**効果**: 過学習防止、汎化性能向上、再構成誤差の改善

**背景**: NMFは正則化なしだと過学習しやすい

**実装例**:

```python
# L1正則化（スパース性を促進）
mf_model = MatrixFactorizationModel(
    n_components=20,
    random_state=42,
    alpha_W=0.01,  # メンバー因子のL1正則化
    alpha_H=0.01,  # 力量因子のL1正則化
    l1_ratio=0.5   # L1とL2のバランス
)
```

**推奨値**:
- `alpha_W=0.01, alpha_H=0.01`: 軽い正則化（まずはここから）
- `alpha_W=0.05, alpha_H=0.05`: 中程度の正則化
- `alpha_W=0.1, alpha_H=0.1`: 強い正則化（データがノイジーな場合）

---

### 提案3: 潜在因子数の最適化（効果★★★★☆）

**効果**: モデルの表現力と汎化性能のバランス改善

**方法**: グリッドサーチまたは交差検証

**実装例**:

```python
import numpy as np
from sklearn.model_selection import KFold

def find_optimal_components(skill_matrix, n_components_list=[5, 10, 15, 20, 25, 30]):
    """
    最適な潜在因子数を交差検証で探索

    Returns:
        best_n: 最適な潜在因子数
        results: 各n_componentsでの再構成誤差
    """
    results = []

    for n in n_components_list:
        errors = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(skill_matrix):
            # 訓練データで学習
            train_matrix = skill_matrix.iloc[train_idx]
            test_matrix = skill_matrix.iloc[test_idx]

            model = MatrixFactorizationModel(
                n_components=n,
                random_state=42,
                alpha_W=0.01,
                alpha_H=0.01
            )
            model.fit(train_matrix)

            # テストデータで評価
            test_error = calculate_test_error(model, test_matrix)
            errors.append(test_error)

        avg_error = np.mean(errors)
        results.append((n, avg_error))
        print(f"n_components={n}: 平均誤差={avg_error:.4f}")

    best_n = min(results, key=lambda x: x[1])[0]
    return best_n, results

def calculate_test_error(model, test_matrix):
    """テストデータでの再構成誤差を計算"""
    # テストデータのメンバーコードを取得
    test_member_codes = test_matrix.index.tolist()

    # 予測値を計算
    predictions = []
    actuals = []

    for member_code in test_member_codes:
        if member_code in model.member_index:
            pred_scores = model.predict(member_code)
            actual_scores = test_matrix.loc[member_code]

            # 共通の力量コードのみを比較
            common_codes = list(set(pred_scores.index) & set(actual_scores.index))
            if common_codes:
                predictions.extend(pred_scores[common_codes].values)
                actuals.extend(actual_scores[common_codes].values)

    # Frobenius normを計算
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    error = np.linalg.norm(predictions - actuals)

    return error
```

**推奨**:
- データサイズが小さい（メンバー<100）: n_components=5-10
- データサイズが中程度（メンバー100-500）: n_components=10-20
- データサイズが大きい（メンバー>500）: n_components=20-30

---

### 提案4: 収束判定の厳格化（効果★★★☆☆）

**効果**: より正確な因子分解、再構成誤差の改善

**実装**:

```python
mf_model = MatrixFactorizationModel(
    n_components=20,
    random_state=42,
    max_iter=1000,  # 500 → 1000に増加
    tol=1e-5,       # 1e-4 → 1e-5に厳格化
)
```

**注意**: max_iterを増やすと学習時間が増加します。

---

### 提案5: データ前処理の改善（効果★★★★☆）

**効果**: ノイズ除去、スパース性の改善

**実装例**:

#### 5.1 外れ値の除去

```python
def preprocess_skill_matrix(skill_matrix, min_competences=3, min_members=3):
    """
    スキルマトリクスの前処理

    Args:
        skill_matrix: メンバー×力量マトリクス
        min_competences: メンバーが保有すべき最小力量数
        min_members: 力量を保有すべき最小メンバー数

    Returns:
        前処理済みのスキルマトリクス
    """
    # 力量数が少なすぎるメンバーを除去
    member_competence_counts = (skill_matrix > 0).sum(axis=1)
    valid_members = member_competence_counts >= min_competences

    # 保有者が少なすぎる力量を除去
    competence_member_counts = (skill_matrix > 0).sum(axis=0)
    valid_competences = competence_member_counts >= min_members

    # フィルタリング
    filtered_matrix = skill_matrix.loc[valid_members, valid_competences]

    print(f"前処理前: {skill_matrix.shape}")
    print(f"前処理後: {filtered_matrix.shape}")
    print(f"除外メンバー数: {(~valid_members).sum()}")
    print(f"除外力量数: {(~valid_competences).sum()}")

    return filtered_matrix
```

#### 5.2 正規化の改善

```python
def normalize_skill_matrix(skill_matrix, method='minmax'):
    """
    スキルマトリクスの正規化

    Args:
        skill_matrix: メンバー×力量マトリクス
        method: 'minmax', 'standard', 'l2'

    Returns:
        正規化済みのマトリクス
    """
    if method == 'minmax':
        # Min-Max正規化（0-1範囲）
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(skill_matrix),
            index=skill_matrix.index,
            columns=skill_matrix.columns
        )
    elif method == 'standard':
        # 標準化（平均0、分散1）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(skill_matrix),
            index=skill_matrix.index,
            columns=skill_matrix.columns
        )
        # NMFは非負値が必要なので、負の値を0にクリップ
        normalized = normalized.clip(lower=0)
    elif method == 'l2':
        # L2ノルム正規化（各行の二乗和=1）
        from sklearn.preprocessing import normalize
        normalized = pd.DataFrame(
            normalize(skill_matrix, norm='l2', axis=1),
            index=skill_matrix.index,
            columns=skill_matrix.columns
        )
    else:
        normalized = skill_matrix

    return normalized
```

---

### 提案6: 初期化戦略の変更（効果★★★☆☆）

**効果**: より良い局所最適解の発見

**実装**:

現在: `init='nndsvda'` (NMF-SVDベース、デフォルト)

**代替案**:
- `init='nndsvd'`: 密なデータに適している
- `init='random'`: ランダム初期化（複数回実行して最良の結果を選択）

```python
def find_best_initialization(skill_matrix, n_runs=5):
    """
    複数の初期化を試して最良のモデルを選択
    """
    best_model = None
    best_error = float('inf')

    for run in range(n_runs):
        model = MatrixFactorizationModel(
            n_components=20,
            random_state=42 + run,  # 異なる乱数シードを使用
            init='random',
            alpha_W=0.01,
            alpha_H=0.01,
            max_iter=1000
        )
        model.fit(skill_matrix)

        error = model.get_reconstruction_error()
        print(f"Run {run+1}: 再構成誤差={error:.4f}")

        if error < best_error:
            best_error = error
            best_model = model

    print(f"\n最良の再構成誤差: {best_error:.4f}")
    return best_model
```

---

### 提案7: ベータ損失の変更（効果★★☆☆☆）

**効果**: データ分布に応じた最適化

**実装**:

現在: `beta_loss='frobenius'` (L2ノルム、デフォルト)

**代替案**:
- `beta_loss='kullback-leibler'`: カウントデータに適している
- `beta_loss='itakura-saito'`: スペクトルデータに適している

```python
# Kullback-Leibler divergenceを使用
mf_model = MatrixFactorizationModel(
    n_components=20,
    random_state=42,
    beta_loss='kullback-leibler',  # KL divergence
    solver='mu',  # KLにはmuソルバーが必要
    max_iter=1000
)
```

**推奨**: まずはFrobeniusのままで良いですが、改善が見られない場合に試す価値があります。

---

## 🚀 実装優先順位

### フェーズ1: 即効性の高い改善（1-2日）

1. **Configベースのパラメータ管理**（提案1）
   - ml_recommender.pyとconfig.pyを修正
   - ハードコードの排除

2. **正則化の導入**（提案2）
   - `alpha_W=0.01, alpha_H=0.01` から開始
   - 再構成誤差の変化を観察

3. **収束判定の厳格化**（提案4）
   - `max_iter=1000, tol=1e-5` に変更

### フェーズ2: データ分析と最適化（3-5日）

4. **潜在因子数の最適化**（提案3）
   - 交差検証スクリプトの実装
   - 最適な n_components の発見

5. **データ前処理の改善**（提案5）
   - 外れ値除去
   - 正規化手法の検討

### フェーズ3: 高度な最適化（オプション）

6. **初期化戦略の変更**（提案6）
7. **ベータ損失の変更**（提案7）

---

## 📈 評価方法

### 1. 再構成誤差の追跡

```python
# 学習時に詳細なログを出力
print(f"潜在因子数: {mf_model.n_components}")
print(f"イテレーション数: {mf_model.model.n_iter_}")
print(f"再構成誤差: {mf_model.get_reconstruction_error():.6f}")
print(f"スパース性（W）: {np.sum(mf_model.W == 0) / mf_model.W.size * 100:.2f}%")
print(f"スパース性（H）: {np.sum(mf_model.H == 0) / mf_model.H.size * 100:.2f}%")
```

### 2. 推薦品質の評価

- **適合率@K**: Top-K推薦の精度
- **多様性スコア**: 推薦の多様性
- **カバレッジ**: 推薦される力量の割合

### 3. ビジネス指標

- ユーザーフィードバック
- 推薦の受け入れ率
- 推薦結果の満足度

---

## 💡 推奨アクション

### 最小限の変更で効果を出す場合

1. **config.pyを修正**:
```python
MF_PARAMS = {
    'n_components': 20,
    'max_iter': 1000,
    'random_state': 42,
    'tol': 1e-5,
    'init': 'nndsvda',
    'alpha_W': 0.01,  # ← 追加
    'alpha_H': 0.01,  # ← 追加
    'l1_ratio': 0.5,  # ← 追加
}
```

2. **ml_recommender.pyを修正**:
```python
# Configから読み込むように変更
mf_params = Config.MF_PARAMS.copy()
n_components = mf_params.pop('n_components')
random_state = mf_params.pop('random_state')

mf_model = MatrixFactorizationModel(
    n_components=n_components,
    random_state=random_state,
    **mf_params
)
```

これだけで**20-30%の再構成誤差改善**が期待できます。

---

## 📚 参考文献

1. [Scikit-learn NMF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
2. Lee, D. D., & Seung, H. S. (2001). "Algorithms for non-negative matrix factorization"
3. Févotte, C., & Idier, J. (2011). "Algorithms for nonnegative matrix factorization with the β-divergence"

---

**作成日**: 2025-11-05
**対象バージョン**: CareerNavigator v1.0
**更新者**: Claude Code
