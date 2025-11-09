# SEM（構造方程式モデリング）統合ガイド

## 概要

**SkillDomainSEMModel**は、CareerNavigatorのハイブリッド推薦システムに統合可能な推薦スコア強化モジュールです。

現在のハイブリッド推薦は **NMF(協調フィルタリング) + グラフベース(RWR) + コンテンツベース** で構成されていますが、SEMを追加することで、**スキル依存性をベースとした因果効果の定量化** が可能になります。

## システムアーキテクチャ

### 現在のハイブリッド推薦システム

```
【推薦パイプライン】

メンバー & スキルデータ
    ↓
┌─────────────────────────────────────────────┐
│   ハイブリッド推薦エンジン                   │
│                                             │
│  グラフベース(RWR)    NMF協調フィルタリング  │
│  重み: 40%            重み: 30%             │
│     ↓                   ↓                   │
│     └───────┬───────────┘                   │
│             │                              │
│    コンテンツベース (職種/等級)             │
│    重み: 30%                                │
│             │                              │
│             ↓                              │
│      スコア統合・再ランキング               │
│             │                              │
└─────────────┼─────────────────────────────┘
              ↓
        推薦スキル TOP-10

実装: skillnote_recommendation/ml/ml_recommender.py
表示: pages/4_Inference.py
```

### SEM統合後のハイブリッド推薦システム（計画中）

```
【推薦パイプライン with SEM】

メンバー & スキルデータ
    ↓
┌──────────────────────────────────────────────────────┐
│    ハイブリッド推薦エンジン with SEM                 │
│                                                      │
│  グラフベース  NMF協調    コンテンツベース           │
│  (RWR)       フィルタ    (職種/等級)                │
│  重み: 0.35   重み: 0.25  重み: 0.20                │
│     ↓           ↓          ↓                        │
│     └─────┬─────┴──────────┘                        │
│           │                                         │
│    ┌──────┴──────────┐                              │
│    │   SEM因果効果   │  ← 新規追加                  │
│    │   重み: 0.20    │                              │
│    └──────┬──────────┘                              │
│           ↓                                         │
│    スコア統合・再ランキング                         │
│           │                                         │
└───────────┼──────────────────────────────────────┘
            ↓
    推薦スキル TOP-10
   （スキル依存性を考慮）

実装予定: MLSEMRecommender
        (skillnote_recommendation/ml/ml_sem_recommender.py)
統合先: pages/2_Model_Training.py
表示: pages/4_Inference.py
```

## SEMの役割

### 現在のハイブリッド推薦の制限

- **NMF**: 抽象的な20個の潜在因子を学習（「何が相手に勧められるか」は分かるが「なぜ」が不明確）
- **グラフベース**: スキル間の直接的な依存関係のみを考慮（段階的な学習パスを表現しきれない）
- **コンテンツベース**: メンバー属性と職種の親和性（スキル習得の段階性なし）

### SEMが追加する価値

| 機能 | 説明 | 実装箇所 |
|-----|------|--------|
| **測定モデル** | スキル → 潜在段階変数の推定 | `_estimate_measurement_model()` |
| **段階的学習パス** | 初級→中級→上級の段階的遷移 | `_classify_skills_by_level()` |
| **因果効果の定量化** | スキル段階間の因果係数を計算 | `_calculate_path_coefficient()` |
| **統計的有意性** | p値・信頼区間で因果関係の信頼度を判定 | `_estimate_structural_model()` |
| **説明可能性** | 「なぜこのスキルを勧めるのか」を因果関係で説明 | `_generate_sem_explanation()` |

### 具体例

```python
【現在のハイブリッド推薦】
> メンバーM001に推薦スキル
1. Java (スコア: 0.82)
2. Webアプリ開発 (スコア: 0.78)
3. SQL (スコア: 0.75)

理由: NMFが「似たメンバーがこれを習得している」

---

【SEM統合後の推薦（期待値）】
> メンバーM001に推薦スキル
1. Java (スコア: 0.82) ← 基本スコア
   → 調整後: 0.84（SEM: 初級プログラミング達成後の段階的推薦）

2. Webアプリ開発 (スコア: 0.78)
   → 調整後: 0.81（SEM: Java習得がWebアプリ開発習得の予測因子, p=0.042）

3. SQL (スコア: 0.75)
   → 調整後: 0.76（SEM: データベース領域の基礎スキル）

理由: 「あなたは初級プログラミング領域の習得が70%まで進みました。
      次のステップとしてWebアプリ開発スキルをお勧めします。」
```

## SEMの使用方法

### 1. SEMモデルの初期化

#### 単独での使用

```python
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel

# SEMモデルを初期化
sem_model = SkillDomainSEMModel(
    member_competence_df=member_competence_df,
    competence_master_df=competence_master_df,
    num_domain_categories=8,  # スキル領域を8個に分類
    confidence_level=0.95      # 95%信頼区間
)

# 推薦スコアを計算
sem_score = sem_model.calculate_sem_score(
    member_code="M001",
    skill_code="C001"
)
# 結果: 0.65（習得確率65%）
```

#### MLSEMRecommenderでの統合使用

```python
from skillnote_recommendation.ml.ml_sem_recommender import MLSEMRecommender

# ハイブリッド推薦 with SEM を構築
recommender = MLSEMRecommender.build(
    member_competence=member_competence_df,
    competence_master=competence_master_df,
    member_master=member_master_df,
    use_preprocessing=True,
    use_tuning=False,
    use_sem=True,           # SEM統合を有効化
    sem_weight=0.20,        # SEM重み: 20%
    num_domain_categories=8 # スキル領域数
)

# 推薦を実施（SEM統合済み）
recommendations = recommender.recommend(
    member_code="M001",
    top_n=10,
    use_sem=True,
    return_explanation=True  # 説明文を含める
)

# 結果例
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec.competence_name}")
    print(f"   スコア: {rec.priority_score:.3f}")
    print(f"   理由: {rec.reason}")
```

### 2. メンバープロファイルの取得

```python
# メンバーの領域別プロファイルを取得
profile = sem_model.get_member_domain_profile("M001")

# 結果例
{
    'プログラミング': {
        'プログラミング_初級': 0.85,   # 初級スキル達成度85%
        'プログラミング_中級': 0.62,   # 中級スキル達成度62%
        'プログラミング_上級': 0.15    # 上級スキル達成度15%
    },
    'データベース': {
        'データベース_初級': 0.30,
        'データベース_中級': 0.05,
        'データベース_上級': 0.00
    }
}

# 用途
# - メンバーの強み・弱みの可視化
# - キャリア開発相談の基礎資料
```

### 3. 領域情報の詳細取得

```python
# 領域の詳細情報を取得（パス係数、統計量含む）
domain_info = sem_model.get_domain_info("プログラミング")

# 結果例
{
    'domain_name': 'プログラミング',
    'num_latent_factors': 3,  # 潜在変数数
    'latent_factors': [
        {
            'name': 'プログラミング_初級',
            'level': 0,
            'num_skills': 5,
            'factor_loadings': {
                'C001': 0.78,  # Python基礎のローディング
                'C002': 0.65,  # Java基礎のローディング
                ...
            }
        },
        ...
    ],
    'path_coefficients': [
        {
            'from': 'プログラミング_初級',
            'to': 'プログラミング_中級',
            'coefficient': 0.68,           # パス係数
            'p_value': 0.0234,             # 統計的有意性
            't_value': 2.43,
            'is_significant': True,        # p < 0.05
            'ci': (0.15, 1.21)            # 95%信頼区間
        },
        ...
    ]
}
```

## Streamlit UIでの統合計画

### ページ構成（予定）

#### 1. pages/2_Model_Training.py
**現状**: MLRecommenderの学習オプション

**計画**: SEMオプションを追加
```python
# UI追加内容
with st.expander("SEM（構造方程式モデリング）", expanded=False):
    use_sem = st.checkbox(
        "SEMモデルを使用",
        value=True,
        help="スキル依存性を考慮した推薦を有効化"
    )

    if use_sem:
        sem_weight = st.slider(
            "SEM重み",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="SEM因果効果をどの程度推薦スコアに反映させるか"
        )

        num_domains = st.number_input(
            "スキル領域数",
            min_value=5,
            max_value=15,
            value=8,
            help="スキルを分類する領域数（5-15推奨）"
        )

# 実装変更
recommender = MLSEMRecommender.build(
    ...,
    use_sem=use_sem,
    sem_weight=sem_weight if use_sem else 0.0,
    num_domain_categories=num_domains if use_sem else 8
)
```

#### 2. pages/4_Inference.py
**現状**: 推薦結果の表示（NMF + グラフベース + コンテンツベース）

**計画**: SEM分析情報を追加表示
```python
# UI追加内容

# 推薦結果の下に「SEM分析」セクションを追加
if hasattr(recommender, 'sem_model') and recommender.sem_model:
    with st.expander("📊 SEM分析（スキル依存性分析）"):
        # メンバープロファイルをレーダーチャートで表示
        profile = recommender.sem_model.get_member_domain_profile(member_code)
        st.write("### 領域別習得度")
        display_sem_profile(profile)

        # 推奨される学習パスを表示
        st.write("### 推奨学習パス")
        display_sem_learning_path(recommender.sem_model, member_code)

        # 領域別の因果効果を表示
        st.write("### スキル依存関係（パス係数）")
        display_path_coefficients(recommender.sem_model)
```

### コード実装予定

**ファイル**: `skillnote_recommendation/utils/visualization.py`

```python
def display_sem_profile(profile: Dict) -> None:
    """メンバーのSEMプロファイルを可視化"""

    # レーダーチャート作成
    # X軸: 領域名
    # Y軸: 潜在変数スコア（初級/中級/上級の平均）

def display_sem_learning_path(sem_model, member_code: str) -> None:
    """推奨学習パスを表示"""

    # 現在位置 → 次段階 → その次へのパスを表示
    # パス係数と有意性も表示

def display_path_coefficients(sem_model) -> None:
    """領域別パス係数を表示"""

    # 表形式で全領域のパス係数を表示
    # 有意性（p値）を色分けで示す
```

## 実装チェックリスト

### Phase 1: SEMモデルの検証（完了 ✅）
- [x] SEM理論に基づいた実装
- [x] 測定モデルの正確な計算
- [x] 構造モデルの統計的推定
- [x] ファクターローディングの実データからの推定
- [x] パス係数の統計検定

### Phase 2: MLSEMRecommenderの実装（完了 ✅）
- [x] MLRecommenderの拡張
- [x] SEMスコアの統合
- [x] 説明文の自動生成
- [x] SEM重みの調整機能

### Phase 3: Streamlit UIへの統合（未実装 🔲）
- [ ] pages/2_Model_Training.pyにSEMオプション追加
- [ ] pages/4_Inference.pyにSEM分析表示追加
- [ ] visualization.pyに可視化関数追加
- [ ] テスト実装・検証

### Phase 4: ドキュメント & テスト（部分完了 ⚠️）
- [x] SEM_MODEL_GUIDE.md
- [x] SEM_INTEGRATION_GUIDE.md（このファイル）
- [ ] テストの追加（特に統合テスト）
- [ ] ユーザーガイドの追加

## 現在の状態と次のステップ

### 実装済み
```
✅ skillnote_recommendation/ml/skill_domain_sem_model.py (703行)
   - 完全なSEM実装
   - 測定モデル、構造モデル、統計検定
   - メンバープロファイル取得

✅ skillnote_recommendation/ml/ml_sem_recommender.py (401行)
   - MLRecommenderとSEMの統合
   - スコア融合ロジック
   - 説明文生成

✅ tests/test_skill_domain_sem_model.py (288行)
   - ユニットテスト18個

✅ docs/SEM_MODEL_GUIDE.md
   - 理論ガイド
   - 統計量の解釈
```

### 未実装
```
🔲 pages/2_Model_Training.pyのUI追加
   - SEMオプション（有効/無効、重み、領域数）

🔲 pages/4_Inference.pyのUI追加
   - SEM分析セクション
   - プロファイル可視化
   - 学習パス表示
   - パス係数表示

🔲 visualization.pyの関数追加
   - レーダーチャート
   - パス図
   - 統計量テーブル

🔲 統合テスト
   - End-to-Endテスト
   - UIテスト
   - 推奨精度検証
```

## 今後の活用シーン

### 短期（1-2ヶ月）
1. Streamlit UIでSEM分析を表示
2. 推薦スコアへのSEM統合
3. ユーザーフィードバック収集

### 中期（2-3ヶ月）
1. キャリアパス因果構造モデルへの拡張
2. 役職別成長パスの可視化
3. 個人別キャリア相談機能

### 長期（3-6ヶ月）
1. メンバー属性相互作用モデルの実装
2. ベイズSEMへの進化
3. 予測精度の継続改善

## 参考資料

- **実装**: `skillnote_recommendation/ml/skill_domain_sem_model.py`
- **統合**: `skillnote_recommendation/ml/ml_sem_recommender.py`
- **ガイド**: `docs/SEM_MODEL_GUIDE.md`
- **ハイブリッド推薦**: `docs/HYBRID_RECOMMENDATION_SYSTEM.md`
- **テスト**: `tests/test_skill_domain_sem_model.py`
