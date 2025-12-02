# スケーラビリティ分析: スキル1000個のSEM実装

## 📊 **問題の規模**

### 現在のデータ規模
- **メンバー数**: 1000人弱
- **スキル数**: 1000前後
- **データポイント**: 約100万件（1000人 × 1000スキル）

### 標準的なSEMの想定
```
通常のSEM研究:
- 変数数: 10-50個
- サンプルサイズ: 200-1000人
- パラメータ数: 50-200個
```

### スキル1000個の場合の計算量

```python
# 共分散行列のサイズ
共分散行列 = 1000 × 1000 = 100万要素
メモリ: 100万 × 8 bytes = 8 MB（まだ許容範囲）

# フルSEMモデルのパラメータ数（仮定）
測定モデル:
  - ファクターローディング: 1000個
  - 誤差分散: 1000個

構造モデル（フルモデルの場合）:
  - パス係数: 最大 1000 × 999 / 2 = 499,500個 ⚠️
  - 潜在変数の共分散: 数千個

合計: 50万パラメータ以上！ ← 計算不可能
```

### 計算時間の見積もり

```python
# 標準的なSEM（50変数）
推定時間: 1-10秒（ML推定、最適化100-500回）

# スキル1000個（ナイーブな実装）
推定時間: 数時間〜数日 ❌

理由:
1. 逆行列計算: O(n³) = 1000³ = 10億回の演算
2. 目的関数の評価: 各イテレーションで重い計算
3. 収束まで数百回のイテレーション
```

---

## ✅ **解決策: 階層的・疎構造モデル**

### アプローチ1: 階層的潜在変数モデル（推奨）

```
レベル3: 総合力量（1-3個）
         │
         ├─ 技術力
         ├─ ビジネス力
         └─ マネジメント力

レベル2: ドメイン別力量（10-30個）
         │
         ├─ Python開発力
         ├─ Web開発力
         ├─ データ分析力
         └─ ...

レベル1: 個別スキル（1000個）
         │
         ├─ Python基礎
         ├─ Django
         ├─ NumPy
         └─ ...
```

**利点**:
- パラメータ数を大幅削減（数千個 → 数百個）
- 計算時間を短縮（数時間 → 数分）
- 解釈が容易

**実装**:
```python
# 段階的推定
Stage 1: スキル → ドメイン力量（10-30個のサブモデル）
Stage 2: ドメイン力量 → 総合力量（1つの統合モデル）

例:
model_python = '''
# Python開発力ドメイン（スキル50個 → 潜在変数1個）
Python開発力 =~ Python基礎 + Django + Flask + NumPy + ...
'''

model_integration = '''
# 総合モデル（潜在変数20個 → 上位潜在変数3個）
技術力 =~ Python開発力 + Web開発力 + データ分析力 + ...
ビジネス力 =~ 要件定義力 + 企画力 + ...

# 構造モデル
技術力 ~ ビジネス力
マネジメント力 ~ 技術力 + ビジネス力
'''
```

---

### アプローチ2: 疎構造モデル（補助的）

```python
# 全ての1000×999/2 = 49万パスではなく
# 事前定義された重要なパスのみを推定

重要なパス（例: 100-500本）:
- 前提スキル → 発展スキル
- 同一ドメイン内の関連スキル
- クリティカルパス

非重要なパス:
- 係数を0に固定（推定しない）
```

---

### アプローチ3: ドメイン分割モデル

```python
# 各ドメインで独立したSEMを推定
# ドメイン間の関係は上位レベルでモデル化

ドメイン別モデル（20-30個）:
- Python開発ドメイン（50スキル）
- Web開発ドメイン（70スキル）
- データ分析ドメイン（60スキル）
- ...

各ドメイン:
- 推定時間: 1-5秒
- パラメータ数: 100-300個
- 並列処理可能 ✅
```

---

## 🎯 **推奨アーキテクチャ**

### 実装戦略

```
┌─────────────────────────────────────────────┐
│ UnifiedSEMEstimator                        │
│ （統一インターフェース）                     │
└─────────────────────────────────────────────┘
              │
              ├─ HierarchicalSEMEstimator
              │  （階層的推定: メイン）
              │
              ├─ DomainSEMEstimator
              │  （ドメイン別推定）
              │
              └─ SparseSEMEstimator
                 （疎構造推定: 補助）
```

### Phase別実装計画

#### **Phase 1: 階層的SEMの実装（1-2週間）**

```python
class HierarchicalSEMEstimator:
    """
    階層的SEM推定器

    3層構造:
    - L1: スキル（1000個）
    - L2: ドメイン力量（20-30個）
    - L3: 総合力量（3-5個）
    """

    def __init__(self, hierarchy: Dict[str, List[str]]):
        """
        hierarchy = {
            'Python開発力': ['Python基礎', 'Django', ...],
            'Web開発力': ['HTML', 'CSS', 'JavaScript', ...],
            ...
        }
        """
        self.hierarchy = hierarchy

    def fit(self, data: pd.DataFrame):
        """段階的推定"""

        # Stage 1: スキル → ドメイン力量
        domain_scores = {}
        for domain, skills in self.hierarchy.items():
            # 各ドメインで独立したSEMを推定
            model = f"{domain} =~ {' + '.join(skills)}"
            sem = semopy.Model(model)
            sem.fit(data[skills])

            # ドメインスコアを計算
            domain_scores[domain] = sem.predict_factors(data[skills])

        # Stage 2: ドメイン力量 → 総合力量
        domain_df = pd.DataFrame(domain_scores)

        integration_model = '''
        技術力 =~ Python開発力 + Web開発力 + ...
        ビジネス力 =~ 要件定義力 + ...

        # 構造モデル（力量同士の関係性）
        技術力 ~ ビジネス力
        マネジメント力 ~ 技術力 + ビジネス力
        '''

        sem_integration = semopy.Model(integration_model)
        sem_integration.fit(domain_df)

        return {
            'domain_models': self.domain_models,
            'integration_model': sem_integration,
            'fit_indices': self._compute_fit_indices()
        }
```

#### **Phase 2: パフォーマンス最適化（3-5日）**

```python
# 並列処理
from joblib import Parallel, delayed

def fit_domain_model(domain, skills, data):
    model = f"{domain} =~ {' + '.join(skills)}"
    sem = semopy.Model(model)
    sem.fit(data[skills])
    return domain, sem

# 20個のドメインを並列処理
results = Parallel(n_jobs=10)(
    delayed(fit_domain_model)(d, s, data)
    for d, s in hierarchy.items()
)
```

#### **Phase 3: 統合と検証（3-5日）**

```python
# 既存モデルとの比較
class SEMModelComparison:
    """旧モデルと新モデルの比較"""

    def compare(self, old_model, new_model, test_data):
        return {
            'fit_indices': self._compare_fit(),
            'prediction_accuracy': self._compare_predictions(),
            'computational_cost': self._compare_performance()
        }
```

---

## 📊 **期待される性能**

### 計算時間の見積もり

```python
# 階層的SEM実装
Stage 1: ドメイン別推定（並列）
  - 20ドメイン × 1-2秒 = 2-4秒（並列10コア）

Stage 2: 統合モデル推定
  - 20変数のSEM = 5-10秒

合計: 7-14秒 ✅

# メモリ使用量
各ドメイン: 1-10 MB
統合モデル: 1 MB
合計: 20-200 MB ✅
```

### パラメータ数

```python
# 階層的モデル
測定モデル:
  - L1→L2: 1000個のローディング
  - L2→L3: 20-30個のローディング

構造モデル:
  - L3レベルのパス: 3-10個

合計: 1050-1040個 ✅（50万 → 1000に削減！）
```

---

## 🔧 **技術スタック**

### 必要なライブラリ

```bash
pip install semopy>=2.3.0
pip install scipy>=1.9.0
pip install numpy>=1.23.0
pip install pandas>=1.5.0
pip install joblib>=1.2.0  # 並列処理
```

### semopyの制限事項

```python
# semopy: 最大100-200変数を推奨
# → ドメイン分割で対応 ✅

# 大規模データでの推定
# → バッチ処理、並列処理で対応 ✅
```

---

## 🎯 **実装ロードマップ（修正版）**

### Week 1: 階層構造の設計と基本実装
- [ ] スキル階層定義の自動生成
- [ ] HierarchicalSEMEstimatorの基本実装
- [ ] 小規模データでの動作確認

### Week 2: ドメイン別推定の実装
- [ ] 各ドメインでのSEM推定
- [ ] ドメインスコアの計算
- [ ] 並列処理の実装

### Week 3: 統合モデルと適合度指標
- [ ] 統合モデルの推定
- [ ] 標準適合度指標の実装
- [ ] 間接効果の計算

### Week 4: 推薦システム統合とテスト
- [ ] 既存システムとの並列運用
- [ ] パフォーマンステスト
- [ ] ドキュメント整備

---

## ✅ **次のステップ**

1. **スキル階層定義の確認**
   - 既存のスキル定義JSONを確認
   - ドメイン分類が存在するか？
   - 自動分類が必要か？

2. **実装開始**
   - HierarchicalSEMEstimatorのプロトタイプ
   - 小規模データでの検証

3. **段階的展開**
   - ドメイン別推定
   - 統合モデル
   - 推薦システム統合
