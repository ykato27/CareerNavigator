# キャリアパス相関構造モデル（Career Path SEM Model）実装ガイド

**バージョン**: v1.1
**最終更新日**: 2025-11-12
**実装**: skillnote_recommendation/ml/career_path_sem_model.py

---

## 目次

1. [概要](#概要)
2. [背景と動機](#背景と動機)
3. [モデルアーキテクチャ](#モデルアーキテクチャ)
4. [実装詳細](#実装詳細)
5. [使用方法](#使用方法)
6. [評価指標](#評価指標)
7. [テスト](#テスト)
8. [参考文献](#参考文献)

---

## 概要

キャリアパス相関構造モデル（Career Path SEM Model）は、**役職ごとの典型的なスキル習得パスを相関構造（パス図）として表現**し、段階間の遷移確率・パス係数を推定することで、メンバーの成長段階に応じた個別化推薦を実現するモデルです。

**重要**: 本モデルは観測データの共分散構造を説明するものであり、因果関係を主張するものではありません。スキル習得の時間順序や因果性を主張するには、追加の仮定（時間順序、除外制約、実験的操作等）が必要です。

### 主な機能

- **成長段階の推定**：メンバーの現在位置をパス上で推定（Stage 0 → Stage 1 → ... → Stage N）
- **段階間遷移確率の計算**：各段階から次の段階への遷移確率とパス係数（相関係数 β）を計算
- **Path Alignment Scoreの計算**：推薦スキルがパスに沿っているかを評価するスコア（0.0～1.0）
- **推薦理由の自動生成**：相関構造に基づく詳細な推薦理由を生成

### 活用シーン

- **キャリア開発コンサルティング**：メンバーの現在位置を可視化し、次のステップを提示
- **学習計画の立案**：個人の成長段階に応じた学習計画を自動生成
- **組織分析**：役職ごとのキャリアパスの標準化と可視化

---

## 背景と動機

### 従来の課題

従来のスキル推薦システムでは、以下の課題がありました：

1. **静的な推薦**：メンバーの成長段階を考慮しない一律の推薦
2. **説明可能性の欠如**：「なぜこのスキルを推薦するのか」が不明確
3. **段階的な成長の無視**：スキル習得の順序や前提条件が考慮されない

### 本モデルの解決策

キャリアパス相関構造モデルは、以下の方法でこれらの課題を解決します：

- **成長段階の可視化**：メンバーがキャリアパスのどの段階にいるかを定量化
- **相関関係の明示**：段階間のパス係数（β）で相関関係の強さを定量化
- **個別化された推薦**：現在の段階と進度に応じて、最適なスキルを推薦

---

## モデルアーキテクチャ

### 全体構造

```
【キャリアパス相関構造SEMモデル】

役職「主任」の例:

  入門期（潜在変数）
    ├─ リーダーシップ基礎（観測スキル）
    ├─ チーム運営（観測スキル）
    └─ 進捗管理（観測スキル）
        ↓ パス係数 β=0.65（相関関係の強さ）
  成長期（潜在変数）
    ├─ プロジェクト管理（観測スキル）
    ├─ リスク管理（観測スキル）
    └─ 目標設定（観測スキル）
        ↓ パス係数 β=0.58（相関関係の強さ）
  熟達期（潜在変数）
    ├─ 複数PJ統括（観測スキル）
    ├─ リソース配分（観測スキル）
    └─ 優先順位決定（観測スキル）
```

### 成長段階（Stage）の定義

各役職について、**3～5段階の成長ステージ**を定義します：

| ステージ | 名称 | 説明 | 典型的期間 |
|---------|------|------|-----------|
| Stage 0 | 入門期 | 基礎スキルの習得 | 3～6ヶ月 |
| Stage 1 | 成長期 | 実践スキルの習得 | 6～12ヶ月 |
| Stage 2 | 熟達期 | 高度なスキルの習得 | 12～24ヶ月 |

### パス係数（Path Coefficient）β

パス係数 β は、**前の段階のスキル習得と次の段階のスキル習得の相関関係の強さ**を表します。

**重要**: これは相関係数であり、因果効果ではありません。前段階のスキル習得が次段階のスキル習得の「原因」であることを証明するものではなく、両者が強く関連していることを示すものです。

- **β = 0.0～0.3**: 相関が弱い（段階間の関連性が低い）
- **β = 0.3～0.6**: 中程度の相関
- **β = 0.6～1.0**: 強い相関（前段階の習得が次段階と強く関連）

---

## 実装詳細

### クラス構造

```python
class CareerPathSEMModel:
    """
    キャリアパス相関構造モデル

    役職ごとの成長パスをSEMとして構造化し、
    段階間の相関関係を定量化する。
    """

    def __init__(
        self,
        member_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        career_path_hierarchy: Optional[CareerPathHierarchy] = None,
    ):
        """初期化"""
        ...

    def fit(
        self,
        roles: Optional[List[str]] = None,
        min_members_per_role: int = 5,
        min_skills_per_stage: int = 3
    ):
        """各役職のキャリアパスSEMモデルを学習"""
        ...

    def calculate_path_alignment_score(
        self,
        member_code: str,
        competence_code: str
    ) -> float:
        """推薦スキルがパスに沿っているかを評価するスコアを計算"""
        ...

    def generate_path_explanation(
        self,
        member_code: str,
        competence_code: str
    ) -> str:
        """推薦理由を生成"""
        ...
```

### 主要メソッド

#### 1. `fit()` - モデル学習

各役職のキャリアパスSEMモデルを学習します。

```python
model = CareerPathSEMModel(
    member_master, member_competence, competence_master
)

# 全役職について学習
model.fit(
    min_members_per_role=5,  # 役職ごとの最低メンバー数
    min_skills_per_stage=3   # ステージごとの最低スキル数
)
```

#### 2. `get_member_position()` - メンバーの現在位置を取得

```python
role, stage, progress = model.get_member_position('M001')

# 出力例:
# role='主任', stage=1, progress=0.65
# → 主任のStage 1（成長期）で、進度65%
```

#### 3. `calculate_path_alignment_score()` - Path Alignment Score計算

推薦スキルがパスに沿っているかを評価します。

```python
score = model.calculate_path_alignment_score('M001', 'C014')

# スコアの解釈:
# 1.0: 現在のステージのスキル（最高優先度）
# 0.8: 次のステージのスキル（高優先度）
# 0.5: 2段階先のスキル（中優先度）
# 0.2: 過去のステージのスキル（低優先度）
# 0.0: パス上にないスキル
```

#### 4. `generate_path_explanation()` - 推薦理由の生成

相関構造に基づく詳細な推薦理由を生成します。

```python
explanation = model.generate_path_explanation('M001', 'C014')

# 出力例:
# 【キャリアパス相関構造モデル推薦】
#
# あなたは現在、役職「主任」の入門期（進度: 65.0%）にいます。
#
# 「プロジェクト管理」は次のステップである成長期のスキルです。
# 現在の段階の進度が65.0%に達しているため、次の段階への準備として推奨します。
#
# 【相関関係】
# 入門期 → 成長期のパス係数: β=0.650
# 現在の段階のスキル習得と次の段階のスキル習得には強い相関関係があります。
#
# Path Alignment Score: 0.85（パス親和性: 85%）
```

#### 5. `recommend_next_steps()` - 次のステップ推薦

メンバーの次のキャリアステップを推薦します。

```python
recommendations = model.recommend_next_steps('M001', top_n=5)

# 出力例:
# [
#   {
#     'competence_code': 'C014',
#     'competence_name': 'プロジェクト管理',
#     'stage': 1,
#     'stage_name': '成長期',
#     'reason': '...',
#     'path_coefficient': 0.65
#   },
#   ...
# ]
```

---

## 使用方法

### 基本的な使用例

```python
import pandas as pd
from skillnote_recommendation.ml.career_path_sem_model import CareerPathSEMModel

# データの読み込み
member_master = pd.read_csv('members.csv')
member_competence = pd.read_csv('member_competence.csv')
competence_master = pd.read_csv('competence_master.csv')

# モデルの初期化と学習
model = CareerPathSEMModel(
    member_master, member_competence, competence_master
)
model.fit(min_members_per_role=5, min_skills_per_stage=3)

# メンバーの現在位置を取得
role, stage, progress = model.get_member_position('M001')
print(f"{role} の Stage {stage} ({progress*100:.1f}%)")

# 次のステップを推薦
recommendations = model.recommend_next_steps('M001', top_n=5)
for rec in recommendations:
    print(f"- {rec['competence_name']} (Path Score: {rec.get('path_coefficient', 0):.2f})")

# 推薦理由を生成
explanation = model.generate_path_explanation('M001', recommendations[0]['competence_code'])
print(explanation)
```

### キャリア進捗サマリーの取得

```python
# メンバーのキャリア進捗サマリーを取得
summary = model.get_career_progression_summary('M001')

print(f"役職: {summary['role']}")
print(f"現在のステージ: {summary['current_stage_name']}")
print(f"進捗率: {summary['progress']*100:.1f}%")
print(f"次のステージ: {summary['next_stage_name']}")
print(f"完了までの推定月数: {summary['estimated_completion_months']}ヶ月")
```

### 役職のキャリアパス全体を可視化

```python
# 役職のキャリアパス全体のサマリーを取得
summary_df = model.get_role_path_summary('主任')
print(summary_df)

# 出力例:
#   Stage  Stage_Name                                 Skills  Path_Coefficient
# 0     0      入門期  リーダーシップ基礎, チーム運営, 進捗管理             N/A
# 1     1      成長期    プロジェクト管理, リスク管理, 目標設定           0.650
# 2     2      熟達期     複数PJ統括, リソース配分, 優先順位           0.580
```

---

## 評価指標

### 1. Path Alignment Score（パス親和性スコア）

推薦スキルがパスに沿っているかを評価する指標：

- **高スコア（0.8～1.0）**: 現在または次のステージのスキル → 優先度が高い
- **中スコア（0.5～0.8）**: 2段階先のスキル → 先行学習として推奨
- **低スコア（0.0～0.5）**: パス上にないまたは過去のスキル → 優先度が低い

### 2. パス係数（Path Coefficient）β

段階間の相関関係を評価する指標：

- **β ≥ 0.6**: 強い相関 → 前段階の習得と次段階の習得が強く関連
- **0.3 ≤ β < 0.6**: 中程度の相関 → 前段階の習得と次段階の習得に一定の関連
- **β < 0.3**: 弱い相関 → 段階間の関連性が低い

### 3. 推薦精度（Recommendation Accuracy）

キャリアパスSEMモデルによる推薦の精度：

- **Precision@5**: 上位5件の推薦のうち、実際に習得されたスキルの割合
- **Recall@5**: メンバーが習得したスキルのうち、推薦に含まれていた割合
- **NDCG@5**: 推薦順位の精度（理想的な順序との一致度）

---

## テスト

### テストファイル

- **tests/test_career_path_sem_model.py**

### テストの実行

```bash
# 全テスト実行
uv run pytest tests/test_career_path_sem_model.py -v

# 特定のテストクラスのみ実行
uv run pytest tests/test_career_path_sem_model.py::TestPathAlignmentScore -v
```

### テストカバレッジ

- モデル学習のテスト
- メンバー位置推定のテスト
- Path Alignment Scoreのテスト
- 推薦理由生成のテスト
- エッジケースのテスト

---

## Streamlit UIへの統合

### 新規ページの作成

キャリアパスSEMモデルを使った推薦ページを作成します：

```python
# pages/8_Career_Path_Recommendation.py

import streamlit as st
from skillnote_recommendation.ml.career_path_sem_model import CareerPathSEMModel

st.title("キャリアパス推薦（SEM相関構造モデル）")

# モデルの初期化
if 'career_path_sem_model' not in st.session_state:
    model = CareerPathSEMModel(
        st.session_state.member_master,
        st.session_state.member_competence,
        st.session_state.competence_master
    )
    model.fit(min_members_per_role=5, min_skills_per_stage=3)
    st.session_state.career_path_sem_model = model

model = st.session_state.career_path_sem_model

# メンバー選択
member_code = st.selectbox("メンバーを選択", st.session_state.member_master['メンバーコード'].tolist())

# 現在位置の表示
role, stage, progress = model.get_member_position(member_code)

st.subheader("現在の位置")
st.metric("役職", role)
st.metric("ステージ", f"Stage {stage}")
st.progress(progress, text=f"進度: {progress*100:.1f}%")

# 推薦の表示
st.subheader("次のステップ推薦")
recommendations = model.recommend_next_steps(member_code, top_n=5)

for rec in recommendations:
    with st.expander(f"{rec['competence_name']} (Path Coefficient: {rec.get('path_coefficient', 0):.2f})"):
        explanation = model.generate_path_explanation(member_code, rec['competence_code'])
        st.write(explanation)
```

---

## 実装箇所

- **モデル実装**: skillnote_recommendation/ml/career_path_sem_model.py
- **階層定義**: skillnote_recommendation/ml/career_path_hierarchy.py
- **テスト**: tests/test_career_path_sem_model.py
- **ドキュメント**: docs/CAREER_PATH_SEM_MODEL.md

---

## 参考文献

1. **SEM（構造方程式モデリング）の基礎**
   - Bollen, K. A. (1989). Structural Equations with Latent Variables. Wiley.

2. **キャリア発達理論**
   - Super, D. E. (1980). A life-span, life-space approach to career development. Journal of Vocational Behavior.

3. **推薦システムへのSEMの応用**
   - Wang, X., & Wang, Y. (2014). Improving content-based and hybrid music recommendation using SEM. Multimedia Systems.

---

## まとめ

キャリアパス相関構造モデルは、**役職ごとのスキル習得パスを相関構造として可視化**し、**メンバーの成長段階に応じた個別化推薦**を実現します。

### 主な利点

- **説明可能性の向上**：相関構造に基づく詳細な推薦理由
- **個別化された推薦**：メンバーの現在位置と進度に応じた推薦
- **キャリア開発支援**：次のステップが明確になることで、学習計画が立てやすい

### 今後の拡張案

- **複数のキャリアパスの提案**：標準パスと代替パスの両方を提示
- **パス逸脱の検出**：標準パスから外れたメンバーの検出と支援
- **動的なパス更新**：新しいデータに基づいてパスを自動更新
- **因果推論への拡張**：時間順序データや操作変数を用いた真の因果関係の検証
