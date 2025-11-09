# Claude Development Rules for CareerNavigator

このドキュメントは、CareerNavigatorプロジェクトをClaudeで開発する際の規則と指針です。

---

## 💎 コード品質・ドキュメント品質の原則

### コード品質の最優先事項
#### 1. **可読性（Readability）**
- **誰が読んでも理解できるコード** を書く
- 「1年後の自分が理解できるか」を基準に
- 変数名・関数名は明確で自己説明的に

```python
# 悪い例：何をしているのか不明確
result = calculate(x, y, 0.7)

# 良い例：目的が明確
diversity_adjusted_score = apply_mmr_reranking(
    relevance_scores=candidate_scores,
    diversity_threshold=0.7
)
```

#### 2. **運用保守性（Maintainability）**
- **変更が容易** なコード設計
- 複数箇所に散在する同じロジックを避ける
- 依存関係を最小化（疎結合）

```python
# 悪い例：複数箇所で重複したロジック
# pages/page1.py
random_state = 42
# pages/page2.py
random_state = 42  # 重複！

# 良い例：設定から一元管理
# config.py
DEFAULT_RANDOM_STATE = 42
# pages/page1.py, pages/page2.py
from config import DEFAULT_RANDOM_STATE
```

#### 3. **テスト可能性（Testability）**
- ロジックを小さな単位に分割
- 依存性注入（DI）で外部依存を注入可能に
- モック化可能な設計

#### 4. **パフォーマンス**
- 大量データ処理時のボトルネックを意識
- 無駄な計算・ループを避ける
- メモリ効率を考慮

---

### ドキュメント品質の最優先事項
#### 1. **正確性（Accuracy）**
- **実装と一致する情報のみ記載**
- 実装後、ドキュメントを必ず検証
- 古い情報は削除または「非推奨」と明記

```markdown
# 悪い例：実装と不一致
## 乱数シード設定
UIから乱数シードを設定できます。
（実装: config.pyでのみ設定可能）

# 良い例：実装を正確に反映
## 乱数シード設定
UIの「チューニング詳細設定」から乱数シードを設定できます。
範囲: 0～2147483647、デフォルト: 42

実装箇所:
- UI: pages/2_Model_Training.py:186-193
- 伝搬: skillnote_recommendation/ml/ml_recommender.py:113
- 使用: skillnote_recommendation/ml/hyperparameter_tuning.py:820
```

#### 2. **具体性（Concreteness）**
- 抽象的な説明ではなく、具体的な例・コード例を含める
- 実装ファイル・行番号を明記
- 実行コマンド、期待される出力を記載

```markdown
# 悪い例：曖昧
テストを実行してください。

# 良い例：具体的
## テストの実行
```bash
# 全テスト実行（カバレッジ付き）
uv run pytest --cov=skillnote_recommendation

# 特定の機能のテストのみ
uv run pytest tests/test_hyperparameter_tuning.py -v

# 期待される出力
# ===================== test session starts ======================
# collected 50 items
# tests/test_hyperparameter_tuning.py::test_nxxx PASSED  [  2%]
```

#### 3. **最新性（Currency）**
- 古い情報は即座に更新
- バージョン情報を含める
- 「最終更新日」を記載（重要なドキュメント）

```markdown
# 悪い例：時代遅れ
v1.0 で実装されました。

# 良い例：最新情報を明記
v1.2.1 (2025-11-09) で実装
- 乱数シード設定機能を追加
- テストカバレッジを30% → 47%に改善
```

#### 4. **トレーサビリティ（Traceability）**
- 実装箇所を明記（ファイル:行番号）
- テストケースへのリンク
- 関連するドキュメントへのリンク

```markdown
# 実装箇所
- **UI定義**: pages/2_Model_Training.py:186-193
- **パラメータ伝搬**: skillnote_recommendation/ml/ml_recommender.py:113
- **使用実装**: skillnote_recommendation/ml/hyperparameter_tuning.py:820

# テスト
- tests/test_hyperparameter_tuning.py::test_custom_random_state
- tests/test_ml_recommender.py::test_build_with_tuning_random_state

# 関連ドキュメント
- docs/ML_TECHNICAL_DETAILS.md (ハイパーパラメータチューニング)
- CONTRIBUTING.md (開発ガイド)
```

---

## 🚀 開発前チェックリスト

### 1. **ブランチ戦略**
- [ ] **mainブランチへの直接pushは禁止** - 常に新しいbranchで開発
- [ ] ブランチ命名規則に従う: `feature/*`, `fix/*`, `refactor/*`, `docs/*`, `test/*`, `chore/*`
- [ ] 開発前に `git pull origin main` で最新に同期

### 2. **実装前の確認**
- [ ] GitHubに機能要件またはissueが存在するか確認
- [ ] 既存の実装や関連コードを検索（重複実装を避ける）
- [ ] ドキュメント（README, docs/*.md）で関連機能を確認
- [ ] テスト設計書（docs/TEST_DESIGN.md）で既存テストケースを確認

### 3. **コードレビュー準備**
- [ ] テストを書いている（テストカバレッジ低下は許容されない）
- [ ] ドキュメントの更新も含める
- [ ] `pyproject.toml`の依存関係が正しく更新されているか

---

## 🤝 批判的意見への対応方針

このプロジェクトでは、**批判的フィードバックを開発の質向上の機会**として捉えます。

### 基本姿勢

#### 1. **批判的意見は貴重**
- 「なぜそう実装したのか？」という質問は、より良い設計を見つける機会
- 批判は個人攻撃ではなく、実装の改善提案として受け止める
- 技術的な議論を歓迎する

#### 2. **批判を受けた時の対応フロー**

```
批判・指摘を受ける
    ↓
① その批判が有効か検討する
    ↓
② 有効な場合：改善案を提示する
    ↓
③ 改善案を実装する
    ↓
④ テスト・ドキュメントを更新する
    ↓
⑤ 新しいコミットまたはPRで対応を報告
```

#### 3. **改善案の提示方法**

批判を受けた際は、以下の形式で改善案を提示します：

```markdown
## 指摘内容
「max_iterをハイパーパラメータ探索の対象にしている」

## 検討結果
- ✅ 指摘は有効：計算コスト効率が悪い
- NMFは高速に収束するため、イテレーション数の増加は性能向上に寄与しない
- Early Stoppingが既に実装されているため、max_iterの探索は不要

## 改善案
1. max_iterを固定値（500）に変更
2. 代わりに正則化係数（alpha_W, alpha_H）を探索対象に変更
3. 実装コストは低く、テストも既存フレームワークで対応可能

## 実装予定
- feature/optimize-max-iter ブランチで実装
- テスト: test_hyperparameter_tuning.py に3つのテストケースを追加
```

### 具体例：過去の指摘と対応

#### 例1: max_iterの過度な探索
**指摘**: 「max_iterを500～1500で探索するのは計算コスト効率が悪い」

**対応**:
- ✅ max_iterをハイパパラ探索対象から削除
- ✅ 固定値（500）に設定
- ✅ claude.mdに「よくある間違い」として記録
- ✅ hyperparameter_tuning.pyに説明コメントを追加

#### 例2: 複数箇所でのデフォルト値定義
**指摘**: 「デフォルト値が pages/2_Model_Training.py と ml_recommender.py の2箇所に定義されている」

**対応**:
- ✅ Configから取得する統一方式に変更
- ✅ `custom_random_state or config.MF_PARAMS["random_state"]` のパターンで統一
- ✅ claude.mdの「パラメータ管理」セクションで推奨方法を記録

---

## 📋 コーディング規約

### Python コード品質基準

#### 型アノテーション（必須）
```python
# 良い例
def build_ml_recommender(
    transformed_data: dict,
    use_preprocessing: bool = True,
    use_tuning: bool = False,
    tuning_n_trials: Optional[int] = None,
) -> MLRecommender:
    ...

# 悪い例（型なし）
def build_ml_recommender(transformed_data, use_preprocessing=True, ...):
    ...
```

#### ドキュメンテーション（必須）
```python
def optimize(self, show_progress_bar: bool = True) -> Tuple[Dict, float]:
    """
    ハイパーパラメータ最適化を実行

    Args:
        show_progress_bar: プログレスバーを表示するか（デフォルト: True）

    Returns:
        Tuple[Dict, float]: (最適パラメータ, 最小再構成誤差)

    Raises:
        ValueError: パラメータが無効な場合
    """
```

#### 命名規則
- **クラス**: PascalCase (`MatrixFactorizationModel`)
- **関数/メソッド**: snake_case (`build_ml_recommender`)
- **定数**: UPPER_SNAKE_CASE (`MAX_ITER`)
- **プライベートメンバ**: 先頭に`_` (`_suggest_params`)

#### Docstring スタイル
- **NumPy/SciPy** スタイルを使用
- Args, Returns, Raises セクションは必須
- 複雑なロジックには説明コメント（`#`）を追加

### Streamlit UIコード

#### UIコンポーネントの整理
```python
# 推奨: セクション分け
with st.expander("⚙️ 学習オプション", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        # UIコンポーネント1
    with col2:
        # UIコンポーネント2
    with col3:
        # UIコンポーネント3（新しく追加する場合）

# 悪い例: UIが横に長くなりすぎ
col1, col2 = st.columns(2)  # 既に3つある場合は非推奨
```

#### UIのデバッグ情報
- 本番コードでデバッグ出力を残さない
- デバッグ用コンテナ（debug_container）を使用
- session_state.debug_messages に記録

#### パラメータのUIコンポーネント化
- UIで設定可能にするパラメータは `st.number_input`, `st.selectbox` などで設定
- デフォルト値は明確に定義（magic number禁止）
- min/max値は理にかなった範囲に設定

### 検証とチェック

```bash
# フォーマット（Black）
uv run black skillnote_recommendation/ pages/

# Lint（Flake8） - 1行100文字まで
uv run flake8 skillnote_recommendation/ pages/ --max-line-length=100

# 型チェック（mypy）
uv run mypy skillnote_recommendation/ pages/
```

---

## 🧪 テスト規約

### テストカバレッジ
- **最小要件**: カバレッジ低下は許容されない（現在: 47%）
- **新規機能**: 80%以上のカバレッジを目指す
- テストなしの機能追加は受け付けない

### テストファイルの配置
```
tests/
├── test_ml_recommender.py       # ML推薦システム
├── test_hyperparameter_tuning.py # ハイパーパラメータチューニング
├── test_matrix_factorization.py  # 行列分解モデル
├── test_knowledge_graph.py       # グラフベース推薦
└── ...
```

### テスト実行
```bash
# すべてのテストを実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=skillnote_recommendation

# 特定のテストファイルのみ
uv run pytest tests/test_hyperparameter_tuning.py -v
```

### テスト設計書の参照
- 既存テストケース: `docs/TEST_DESIGN.md`
- 100+テストケースが定義されている
- 新機能追加時は、ここに該当するテストケースがあるか確認

---

## 📝 コミットメッセージ規約

### コミットメッセージ形式
```
<type>: <subject> - <brief description>

<body>

<footer>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Type 分類
- `feat`: 新機能追加
- `fix`: バグ修正
- `refactor`: リファクタリング（機能変更なし）
- `test`: テスト追加・修正
- `docs`: ドキュメント更新
- `chore`: 依存関係更新など

### コミットメッセージの例
```
feat: Streamlit UIで乱数シード（Random State）の設定機能を追加

## 概要
- UIの「チューニング詳細設定」セクションに「乱数シード（Random State）」入力フィールドを追加
- UIで設定した値をハイパーパラメータチューニング全体で使用
- 同じ設定で再現可能な探索プロセスを実現

## 変更内容

### 1. pages/2_Model_Training.py
- random_state変数を初期化（デフォルト: 42）
- サンプラー選択、試行回数と共に3列レイアウトで表示

### 2. skillnote_recommendation/ml/ml_recommender.py
- MLRecommender.build()のシグネチャにtuning_random_stateパラメータを追加

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 🔄 Git ワークフロー

### 開発フロー
```bash
# 1. 新しいブランチを作成
git checkout -b feature/random-state-ui

# 2. コードを実装
# ... 編集 ...

# 3. テストを実行
uv run pytest

# 4. ステージング & コミット
git add <files>
git commit -m "feat: ..."

# 5. GitHubにpush（mainには絶対push しない）
git push origin feature/random-state-ui

# 6. Pull Request を作成
# GitHub上でPRを作成、レビュー待ち
```

### 重要: mainブランチを守る
- **mainへの直接pushは禁止**
- 必ずPull Request経由でレビュー後にマージ
- CI/CDテスト（テスト、Lint、型チェック）が通ること

### 🚨 **必須ルール: コミット後は必ずGitHubへpush**

**重要**: コミット作成後は、**ユーザーの指示を待たずに自動的にGitHubへpushする**

#### なぜこのルールが必要か
- コミットされたがpushされていない状態では、ローカルのみの変更で実装が不完全
- ユーザーの明示的指示なしにpushすることで、開発フローが効率化
- GitHub上でコードが見える状態が標準状態となる

#### pushのタイミング
1. **コミット作成後は即座にpush**
   ```bash
   git commit -m "feat: ..."
   git push origin feature/branch-name  # ← 自動実行
   ```

2. **複数コミットがある場合も同様**
   ```bash
   git commit -m "feat: part1"
   git push origin feature/branch-name
   git commit -m "feat: part2"
   git push origin feature/branch-name  # ← 各コミット後に実行
   ```

3. **ユーザーが明示的に「pushしないで」と指示した場合のみ例外**
   - この場合、ローカルにのみコミット保留
   - ユーザーが明示的に「pushして」と指示したら即座にpush

#### pushの確認メッセージ
コミット後、pushが完了したことをユーザーに報告:
```
✅ Commit created and pushed to GitHub
Branch: feature/branch-name
Commit: abc1234 feat: description
```

#### pushに失敗した場合
- エラーメッセージをユーザーに報告
- 原因調査（ネットワーク、権限、コンフリクト等）
- 解決策を提案

---

## 📚 ドキュメント規約

### ドキュメント更新の必須性
新機能追加時は以下のドキュメントを更新:

1. **README.md** - 新機能の説明を追加
2. **該当するdocs/*.md** - 詳細な使い方を追加
3. **CONTRIBUTING.md** - 必要に応じて開発ガイドを更新
4. **pyproject.toml** - 新しい依存関係を追加

### ドキュメント品質
- **日本語** で記述
- **マークダウン形式** を使用
- **実装例を含める** - コード例は必須
- **目次を含める** - 見出しレベルを適切に

### 更新例
```markdown
## 乱数シード（Random State）設定機能

ハイパーパラメータチューニング時に乱数シードをUIから設定できるようになりました。

### 使い方
1. Streamlitアプリを起動
2. 「モデル学習」ページで「ハイパーパラメータチューニング」を有効化
3. チューニング詳細設定で「乱数シード」を入力

### 利点
- **再現性**: 同じシードで同じ探索プロセスを再現可能
- **実験**: 異なるハイパーパラメータで同じ初期条件での比較が可能
```

---

## 🎯 機能設計の指針

### 設計時に確認すべき点

#### 1. **スコープの明確化**
- 機能の責務を1つに絞る
- 関連する複数の変更がある場合はリファクタリング検討

#### 2. **既存実装の活用**
- 既存コード・ライブラリの重複実装を避ける
- 同じことを複数回実装しない

#### 3. **UIとバックエンドの分離**
```
✅ 推奨
- UI層（pages/*.py）: ユーザー入力受け取り、結果表示
- 中間層（build_ml_recommender）: パラメータ受け取り、呼び出し
- 実装層（ml_recommender.py）: 実装、戻り値

❌ 非推奨
- UIからビジネスロジックを直接呼び出し
- パラメータのデフォルト値が複数箇所に存在
```

#### 4. **パラメータ管理**
```python
# 良い例: UIからの入力をそのまま渡す
random_state = st.number_input("乱数シード", value=42)
build_ml_recommender(..., tuning_random_state=int(random_state))

# 悪い例: UIで定義、中間層で再定義
random_state_ui = st.number_input(..., value=42)
random_state_default = 42  # ここで重複定義
```

#### 5. **初期化順序**
```python
# 良い例: 先に変数を初期化
random_state = 42
custom_search_space = None

# UIで定義
with st.expander(...):
    random_state = st.number_input(...)
    # ...

# 使用
build_ml_recommender(..., tuning_random_state=random_state)

# 悪い例: 初期化なしで使用
# random_state が未定義の可能性
build_ml_recommender(..., tuning_random_state=random_state)
```

---

## 🚨 よくある間違い

### 1. **maxiterをハイパーパラメータとして探索**
❌ **避けるべき**
- 計算コスト効率が悪い（500-1500で最大3倍時間）
- NMFは高速に収束するため効果が薄い
- Early Stoppingが既に実装されている

✅ **推奨**
- 固定値に設定（例: 500）
- 代わりに正則化係数（alpha_W, alpha_H）を探索

### 2. **パラメータのmagic number**
❌ 非推奨
```python
n_trials = 50  # なぜ50？
```

✅ 推奨
```python
# UIで設定可能 かつ ドキュメント付き
n_trials = st.number_input(
    "試行回数",
    min_value=10,
    max_value=200,
    value=50,
    help="探索する組み合わせの数..."
)
```

### 3. **デバッグコードを残す**
❌ 本番コード
```python
print("[DEBUG] チューニング開始...")
print(f"[DEBUG] n_trials: {n_trials}")
```

✅ デバッグコンテナ使用
```python
debug_messages.append(f"[DEBUG] チューニング開始...")
debug_info.code("\n".join(debug_messages))
```

### 4. **複数箇所でのデフォルト値定義**
❌ 危険
```python
# pages/2_Model_Training.py
n_trials = 50

# ml_recommender.py
if custom_n_trials is None:
    n_trials = 50  # 重複定義！
```

✅ 推奨
```python
# pages/2_Model_Training.py
n_trials = st.number_input(..., value=50)

# ml_recommender.py
if custom_n_trials is None:
    n_trials = optuna_params["n_trials"]  # Configから取得
```

---

## ✅ Pull Request チェックリスト

### PRを作成する前に
- [ ] テストが全て通っているか: `uv run pytest`
- [ ] カバレッジが低下していないか: `uv run pytest --cov=...`
- [ ] Lint が通っているか: `uv run flake8`
- [ ] 型チェックが通っているか: `uv run mypy`
- [ ] フォーマットが正しいか: `uv run black --check`
- [ ] ドキュメント更新は含まれているか
- [ ] コミットメッセージが規約に従っているか

### PR作成時
- [ ] タイトルは `feat:`, `fix:` などで始まっているか
- [ ] 説明に「何を」「なぜ」が含まれているか
- [ ] テストコード・ドキュメントが含まれているか
- [ ] mainブランチをベースにしているか

### マージ前
- [ ] Reviewが approve されているか
- [ ] CI/CDテストが全て通っているか
- [ ] コンフリクトが解決されているか

---

## 🔍 レビューポイント

### コードレビュー時に確認すること

#### A. **可読性チェック**
- [ ] 変数名・関数名は自己説明的か？
  ```python
  # 悪い例
  result = process(data, 0.7)

  # 良い例
  filtered_recommendations = apply_diversity_reranking(
      candidate_list=recommendations,
      diversity_weight=0.7
  )
  ```
- [ ] ネストの深さは3階層以下か？（深すぎると読みづらい）
- [ ] 1関数は50行以下か？（長すぎると理解困難）
- [ ] コメントは「なぜ」を説明しているか？（「何を」ではなく）

#### B. **運用保守性チェック**
- [ ] 同じロジックが複数箇所に存在しないか？
- [ ] 設定値・定数がConfigから取得されているか？
- [ ] 外部依存は最小化されているか？（疎結合）
- [ ] 変更が1箇所で完結するか？（複数箇所修正が必要なら設計を見直す）

#### C. **テスト可能性チェック**
- [ ] ロジックが小さな単位（関数）に分割されているか？
- [ ] 外部依存（DB、ネットワーク等）を注入可能か？
- [ ] 純粋関数（同じ入力で必ず同じ出力）になっているか？

#### D. **ドキュメント正確性チェック**
- [ ] ドキュメントと実装が一致しているか？
- [ ] 実装ファイル・行番号は正確か？
- [ ] 古い情報がないか？（バージョンが古い、非推奨メソッドなど）
- [ ] 実行結果・期待される出力は記載されているか？

#### E. **その他**
1. **型アノテーション**: すべての関数に型情報があるか
2. **テスト**: 新機能に対するテスト（80%以上カバレッジ）があるか
3. **パフォーマンス**: 計算量に問題がないか（大量データでボトルネックなし）

---

## 🛠️ 実装時のチェックリスト（完成前の確認）

### コード実装完了時
```
□ コードが完成した
  ↓
□ 以下の確認を実施
  □ A. 可読性確認
    □ 変数名は明確か？
    □ 関数は1つの責任に絞れているか？
    □ ネストは深くないか？（3階層以下）
    □ 複雑な処理にはコメント（「なぜ」）があるか？

  □ B. 運用保守性確認
    □ 重複したロジックがないか？
    □ 設定は一元管理されているか？
    □ 変更が1箇所で完結するか？

  □ C. テスト
    □ ユニットテストを書いたか？
    □ 80%以上のカバレッジを達成したか？
    □ テストは通るか？ (uv run pytest)

  □ D. ドキュメント
    □ ドキュメントを書いたか？
    □ 実装箇所（ファイル:行番号）を記載したか？
    □ 実装と一致しているか？（実装後に再確認）
    □ 古い情報がないか？

□ 検証を通過した
  ↓
□ コミット・PR作成
```

### チェックリスト実行時の詳細

#### A. 可読性の実装確認
```python
# 実装したコード例
random_state = st.number_input(
    "乱数シード（Random State）",
    min_value=0,
    max_value=2147483647,
    value=42,
    step=1,
    help="乱数シードを固定することで、同じ探索過程を再現できます。実験の再現性が必要な場合に使用します。"
)

✅ 確認項目
- 変数名: random_state ← 意味が明確
- 関数: st.number_input ← 何をしているか明確
- 入力制約: min_value, max_value, value ← 理由が明確（乱数シード范囲）
- ヘルプテキスト: 「なぜ」これが必要か説明
```

#### B. 運用保守性の実装確認
```python
# チェック項目

# ❌ 悪い例：複数箇所で定義
# pages/page1.py
DEFAULT_RANDOM_STATE = 42
# skillnote_recommendation/ml/ml_recommender.py
if custom_random_state is None:
    custom_random_state = 42  # 重複！

# ✅ 良い例：一元管理
# config.py
OPTUNA_PARAMS = {
    "random_state": 42,
    ...
}
# 使用側
random_state = custom_random_state or config.OPTUNA_PARAMS["random_state"]
```

#### C. テスト作成確認
```bash
# 実装時のテスト実行チェック
$ uv run pytest --cov=skillnote_recommendation

# 期待される出力例
collected 393 items
tests/test_ml_recommender.py::test_build_with_tuning_random_state PASSED
tests/test_hyperparameter_tuning.py::test_custom_random_state PASSED
...
======================== 393 passed in 12.34s ========================
============ coverage: 47% ============
```

#### D. ドキュメント確認
```markdown
# 記載すべき内容

## 機能名
乱数シード（Random State）設定機能

## 説明
- 何ができるか
- なぜ必要か
- どう使うか

## 実装箇所
- **UI定義**: pages/2_Model_Training.py:186-193
- **パラメータ伝搬**: skillnote_recommendation/ml/ml_recommender.py:113-122
- **実装**: skillnote_recommendation/ml/hyperparameter_tuning.py:820

## テスト
- tests/test_hyperparameter_tuning.py::test_custom_random_state
- tests/test_ml_recommender.py::test_build_with_tuning_random_state

## 使用例
```python
random_state = 42
recommender = MLRecommender.build(
    ...,
    tuning_random_state=random_state,
    ...
)
```

## 検証方法
✅ ドキュメントを書き終わった直後に実装と照合
✅ 実装ファイル・行番号は正確か
✅ コード例は動作するか
✅ 古い情報がないか
```

---

## 📞 その他の指針

### 質問や判断に迷ったときは
1. README.mdの関連セクションを確認
2. docs/*.md で似た機能を探す
3. CONTRIBUTING.md を参照
4. 既存テストケース（docs/TEST_DESIGN.md）を確認
5. 過去のコミット（git log）で類似実装を参照

### 依存関係の追加
- パッケージ追加前に、既存パッケージで対応できないか確認
- 追加する場合は `uv add` で自動的に `pyproject.toml` を更新
- 開発用のみの場合は `uv add --dev`

---

## 参考リンク

- **README.md**: プロジェクト概要、環境構築
- **CONTRIBUTING.md**: コントリビューションガイド（Git, PR等）
- **docs/TEST_DESIGN.md**: テスト設計書（100+テストケース）
- **docs/CODE_STRUCTURE.md**: コード構造ガイド
- **docs/MODELS_TECHNICAL_GUIDE.md**: ML技術解説（初級者向け）
- **skillnote_recommendation/core/README.md**: コアモジュール説明
- **skillnote_recommendation/ml/README.md**: ML モジュール説明

