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

## 📁 フォルダ構成とファイル配置規約

### 必須ルール: ファイルは適切なディレクトリに配置する

#### テストファイル（test_*.py）
**ルール**: すべてのテストファイルは `tests/` ディレクトリに配置する

```
✅ 正しい配置
tests/
├── test_ml_recommender.py
├── test_hyperparameter_tuning.py
├── test_unified_sem_estimator.py
└── test_hierarchical_sem.py

❌ 誤った配置
/CareerNavigator/test_hierarchical_sem.py        # ルートディレクトリにテストファイル
/CareerNavigator/skillnote_recommendation/graph/test_knowledge_graph.py  # 実装ディレクトリにテストファイル
```

**理由**:
- テストの一元管理: すべてのテストが1箇所に集約され、管理しやすい
- pytest実行の効率化: `pytest tests/` で全テストを実行可能
- CI/CDとの統合: テストディレクトリが明確に定義されている

#### ドキュメントファイル（*.md）
**ルール**: ルートには最小限のドキュメントのみ配置し、その他はすべてdocs/に配置する

```
✅ 正しい配置
/CareerNavigator/
├── README.md                          # プロジェクト概要（ルート・必須）
├── CONTRIBUTING.md                    # コントリビューションガイド（ルート・推奨）
└── docs/                              # すべての詳細ドキュメント
    ├── ARCHITECTURE.md                # アーキテクチャ概要
    ├── API_REFERENCE.md               # API リファレンス
    ├── REFACTORING_GUIDE.md           # リファクタリングガイド
    ├── SEM_SCALABILITY_ANALYSIS.md    # SEM スケーラビリティ分析
    ├── STREAMLIT_CLOUD_SETUP.md       # Streamlit Cloud セットアップ
    ├── SEM_IMPLEMENTATION_SUMMARY.md  # SEM実装サマリー
    ├── TESTING_QUICKSTART.md          # テストガイド
    └── CODE_STRUCTURE.md              # コード構造

❌ 誤った配置
/CareerNavigator/ARCHITECTURE.md       # アーキテクチャドキュメントはdocs/へ
/CareerNavigator/MY_NOTES.md           # 個人メモはルートに配置しない
/CareerNavigator/GUIDE.md              # ガイドはdocs/へ
```

**ドキュメントの分類**:
- **ルート直下**: README.md と CONTRIBUTING.md のみ（GitHub標準）
- **docs/**: すべての技術ドキュメント、ガイド、チュートリアル、リファレンス
- **skillnote_recommendation/*/README.md**: 各モジュールの説明

#### 一時ファイル・デバッグファイル
**ルール**: 一時ファイルやデバッグ用ファイルはルートディレクトリに残さない

```
❌ 削除すべきファイル
/CareerNavigator/test_debug.py         # デバッグ用スクリプト
/CareerNavigator/scratch.py            # 試作コード
/CareerNavigator/sem_code_examples.py  # サンプルコード
/CareerNavigator/temp_*.py             # 一時ファイル

✅ 適切な対応
- デバッグコード: 実装完了後に削除
- サンプルコード: docs/examples/ に移動
- 一時ファイル: 作業完了後に削除
```

**理由**:
- プロジェクトの整理: ルートディレクトリがすっきりし、重要なファイルが見つけやすい
- 混乱の防止: 本番コードと一時コードが混在しない
- Gitの汚染防止: 不要なファイルがコミットされない

#### Pythonスクリプト（*.py）
**ルール**: Pythonファイルは機能に応じた適切なディレクトリに配置する

```
✅ 正しい配置
/CareerNavigator/
├── streamlit_app.py                   # メインエントリーポイント（ルート可）
├── pages/                             # Streamlitページ
│   ├── 1_Data_Loading.py
│   └── 2_Model_Training.py
├── skillnote_recommendation/          # コアライブラリ
│   ├── ml/
│   │   ├── ml_recommender.py
│   │   └── unified_sem_estimator.py
│   └── core/
│       └── data_loader.py
└── tests/                             # テストコード
    └── test_ml_recommender.py

❌ 誤った配置
/CareerNavigator/my_script.py          # スクリプトはルートに配置しない
/CareerNavigator/utils.py              # ユーティリティはskillnote_recommendation/utils/へ
```

#### ルートディレクトリに配置して良いファイル
以下のファイルのみルートディレクトリに配置可能:

```
✅ 許可されるルートファイル
/CareerNavigator/
├── streamlit_app.py          # アプリケーションエントリーポイント
├── pyproject.toml            # 依存関係管理
├── README.md                 # プロジェクト概要（必須）
├── CONTRIBUTING.md           # コントリビューションガイド（推奨）
├── .gitignore                # Git設定
├── .python-version           # Pythonバージョン指定
└── uv.lock                   # uvロックファイル

❌ ルートに配置してはいけないもの
├── ARCHITECTURE.md           # → docs/ に移動
├── API_REFERENCE.md          # → docs/ に移動
├── *_GUIDE.md                # → docs/ に移動
└── その他の技術ドキュメント  # → docs/ に移動
```

### ファイル配置チェックリスト

**新規ファイルを作成する前に**:
- [ ] このファイルは本当に必要か？（一時的なデバッグコードではないか）
- [ ] 適切なディレクトリに配置しているか？
- [ ] ルートディレクトリに配置する正当な理由があるか？

**実装完了時に**:
- [ ] テストファイル（test_*.py）は `tests/` にあるか？
- [ ] 一時ファイル・デバッグファイルを削除したか？
- [ ] ルートディレクトリに不要なファイルが残っていないか？
- [ ] ドキュメントは適切な場所に配置されているか？

**コミット前に**:
```bash
# ルートディレクトリのPythonファイルを確認
ls *.py

# 許可されていないファイルがあれば削除または移動
# 例: test_*.py → tests/ に移動
# 例: debug_*.py → 削除
```

### フォルダ構成の確認方法

```bash
# プロジェクト構造を確認
tree -L 2 -I '__pycache__|*.pyc|.git'

# テストファイルの配置を確認
find . -name "test_*.py" -o -name "*_test.py"

# ルートディレクトリのPythonファイルを確認
ls -la /*.py
```

### 違反時の対応

**フォルダ構成ルール違反を発見した場合**:
1. 即座に適切な場所に移動
2. Gitで変更をコミット
3. 今後同じ間違いをしないようclaude.mdを参照

```bash
# 例: ルートのテストファイルをtests/に移動
mv /CareerNavigator/test_hierarchical_sem.py /CareerNavigator/tests/test_hierarchical_sem.py

# コミット
git add tests/test_hierarchical_sem.py
git commit -m "refactor: ルートのテストファイルをtests/に移動"
git push
```

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

---

## 🔬 スキル依存性SEMモデル（構造方程式モデリング）の活用アイディア

### 概要
現在のCareerNavigatorには、スキル依存性を検出する`SkillDependencyAnalyzer`が実装されており、
これをSEM（構造方程式モデリング）ベースで拡張することで、より深い因果関係の分析と推薦が可能になります。

### 現在の実装状況
- ✅ 単一ステップの依存性検出（A → B）
- ✅ 信頼度スコアリング（0.3～1.0）
- ✅ 依存性強度の分類（強/中/弱/なし）
- ✅ 並列習得可能スキルの検出
- ✅ 役職別スキル習得パスの分析
- ❌ マルチステップ依存チェーン（A→B→C）
- ❌ 条件付き依存性
- ❌ 因果効果の定量化

### 活用アイディア3つ

#### 1️⃣ スキル領域潜在変数モデル（推奨度 ⭐⭐⭐⭐⭐）

**概念**
スキルを5～10個のカテゴリに分類し、各カテゴリ内に「初級スキル」「中級スキル」「上級スキル」
という潜在変数を設定。観測可能なスキル習得レベルから潜在変数を推定し、
スキル間の構造的な依存関係を把握します。

**具体例**
```
【プログラミング領域の構造】
    ↓
初級プログラミング（潜在変数）
    ├─ Python基礎（観測スキル）
    ├─ Java基礎（観測スキル）
    └─ Git（観測スキル）
    ↓
中級プログラミング（潜在変数）
    ├─ Webアプリ開発（観測スキル）
    ├─ マイクロサービス（観測スキル）
    └─ ユニットテスト（観測スキル）
    ↓
上級プログラミング（潜在変数）
    ├─ システム設計（観測スキル）
    └─ アーキテクチャ設計（観測スキル）

【データベース領域の構造】
初級DB → 中級DB → 上級DB
```

**実装ステップ**
1. `skillnote_recommendation/ml/skill_sem_model.py` を新規作成
2. スキルカテゴリを5～10個に集約（`category_hierarchy.py`を活用）
3. 各カテゴリで3～4レベルの潜在変数を定義
4. メンバーのスキル習得レベルから潜在変数の推定値を計算
5. スキル間の構造的パス（直接効果・間接効果）を推定

**推薦への活用**
```python
# 例: target_memberがPython基礎を習得している場合
sem_model = SkillSEMModel(member_competence_df, skill_categories)

# 直接効果：「初級プログラミング」から推奨されるスキル
direct_recommendations = sem_model.get_direct_effect_skills(
    member_code="M001",
    skill_category="プログラミング"
)
# 結果: [Webアプリ開発, マイクロサービス, ユニットテスト]

# 間接効果：他の領域スキルの習得を助けるスキル
indirect_support = sem_model.get_indirect_support_skills(
    target_skill="システム設計",
    member_code="M001"
)
# 結果: [DB設計, アーキテクチャ思考, デザインパターン]

# スコアに組み込む
recommendation_score = (
    0.3 * nmf_score +
    0.3 * graph_score +
    0.2 * content_score +
    0.2 * sem_direct_effect_score  # SEM新規
)
```

**メリット**
- NMFの20個の抽象的潜在因子より、実務的に理解しやすい
- 「まずこれを習得してから」という前提条件が明確
- 推薦理由の説明が具体的（「初級プログラミングスキルを強化するため」など）
- マルチステップ推薦が可能（A→B→Cの段階的推薦）

#### 2️⃣ キャリアパス因果構造モデル（推奨度 ⭐⭐⭐⭐⭐）

**概念**
役職ごとの典型的なスキル習得パスを、因果構造（パス図）として表現します。
現在の`RoleBasedGrowthPathAnalyzer`を拡張して、
「なぜこの順序でスキルを習得するのか」の因果メカニズムを可視化・定量化します。

**具体例**
```
【営業 → 営業管理職への成長パス】

段階0（初期）
└─ 営業基礎スキル（顧客対応、提案資料作成）

↓ 営業経験の蓄積（3年程度）
↓
↓ 「営業経験」→「コミュニケーション力向上」（因果係数 β=0.65）
↓
段階1（中期）
└─ マネジメント基礎（面談スキル、フィードバック、目標設定）

↓ マネジメント経験の蓄積（2年程度）
↓
↓ 「マネジメント経験」→「戦略思考力向上」（因果係数 β=0.58）
↓
段階2（後期）
└─ 意思決定スキル（予算管理、事業計画、リスク管理）
```

**実装ステップ**
1. 各役職のスキル習得パスを3～5段階に分割（現在のデータから学習）
2. 段階間の遷移確率を計算
3. パス係数（因果係数）を推定
   ```python
   # 段階0→1への遷移確率
   transition_prob = (
       num_members_advancing_to_stage1 /
       num_members_completing_stage0
   )

   # 段階0スキル習得の「深さ」が段階1進出に与える影響
   path_coefficient = calculate_sem_path_coefficient(
       stage0_skill_depth,
       stage1_entry_probability
   )
   ```
4. メンバーの現在の進度をパス図上にプロット
5. 推薦システムに「パスに従う確率」を組み込む

**推薦への活用**
```python
# 例: M001は「主任」役職で、段階1（マネジメント基礎）学習中
growth_path_model = CareerPathSEMModel(
    member_competence_df,
    role_growth_paths
)

# メンバーの現在位置
current_stage = growth_path_model.estimate_member_stage("M001")
# 結果: Stage 1 (進度: 60%)

# 推奨される次のスキル（パス上で次に習得すべき）
next_skills = growth_path_model.recommend_next_on_path(
    member_code="M001",
    top_n=5
)
# 結果: [事業計画スキル, 予算管理, リスク管理, ...]

# 推薦理由の生成
explanation = growth_path_model.generate_path_explanation("M001")
# 出力: 「主任職の標準的成長パス上で、次のステップとして事業計画スキルが重要です。
#       あなたのマネジメント基礎スキル（進度60%）の次段階としておすすめです。」

# スコアに組み込む
path_alignment_score = sem_model.calculate_path_alignment(
    member_code="M001",
    recommended_skill="事業計画スキル"
)
# 結果: 0.78（パスとの親和性78%）
```

**メリット**
- 同じ役職でも異なるキャリアパスを取ったメンバーを説明可能
- 「標準パスからの逸脱」が有効な代替パスかどうか判定可能
- 次のステップが「推定」されるため、個別化された推薦が可能
- キャリア開発コンサルティング機能として活用可能

#### 3️⃣ メンバー属性相互作用モデル（推奨度 ⭐⭐⭐）

**概念**
メンバーの属性（経験年数、役職、学歴など）とスキル習得確率の関係を構造化します。
「基礎力量」という潜在変数を導入し、属性がこれに与える影響、
および基礎力量がスキル習得に与える影響を同時にモデル化します。

**具体例**
```
【メンバー属性 → スキル習得確率の構造】

メンバー属性（観測可能）
├─ 経験年数
├─ 役職レベル
└─ 学歴

    ↓
    ↓ SEMパス係数で定量化
    ↓

基礎力量（潜在変数・観測不可）
    │
    ├─→（直接効果 β=0.45） Python習得確率 ↗
    │
    ├─→（直接効果 β=0.38） SQL習得確率 ↗
    │
    └─→（直接効果 β=0.52） システム設計習得確率 ↗

※経験年数が長い人ほど基礎力量が高く、各スキル習得確率が上昇
```

**実装ステップ**
1. `SkillSEMInteractionModel`クラスを新規作成
2. メンバーの観測可能な属性（経験年数、役職、学歴等）を収集
3. 隠れた「基礎力量」因子を定義
4. 属性→基礎力量→スキル習得のパス係数を推定
5. 新規メンバー・未採用スキルに対する習得確率を予測

**推薦への活用**
```python
# 例: M002は経験浅いジュニア、Pythonはまだ習得していない
interaction_model = SkillSEMInteractionModel(
    member_competence_df,
    member_attributes_df
)

# メンバーの「基礎力量」スコアを推定
foundation_score = interaction_model.estimate_foundation_skill(
    member_code="M002"
)
# 結果: 0.35（基礎力量が低い）

# 基礎力量が低い場合でも習得可能なスキルを推奨
foundation_aware_recs = interaction_model.recommend_by_foundation(
    member_code="M002",
    foundation_score=0.35,
    top_n=5
)
# 結果: [Excel VBA, データ分析基礎, Git, ...]
# （Pythonより習得確率が高い）

# 習得確率の予測
acquisition_probability = interaction_model.predict_acquisition_probability(
    member_code="M002",
    skill_code="Python",
    months=6
)
# 結果: 0.42（6ヶ月以内にPython習得する確率42%）

# スコアに組み込む
adjusted_score = original_score * (1 + foundation_score * 0.3)
```

**メリット**
- メンバーの属性に合わせた個別化推薦が可能
- 「このメンバーにはこのスキルは早すぎる」という判定が自動化
- 学習支援プログラムのマッチング精度向上
- キャリアパス多様性の理解

### 実装優先度と期間

| 優先度 | アイディア | 推奨期間 | 期待効果 |
|--------|----------|--------|---------|
| **1位** | スキル領域潜在変数 | 4～6週 | 推薦理由の説明可能性向上 |
| **2位** | キャリアパス因果構造 | 4～6週 | 個別化推薦の精度向上 |
| **3位** | メンバー属性相互作用 | 3～4週 | 属性ベース個別化 |

### 導入チェックリスト

**Before実装**
- [ ] メンバーの取得日データが70%以上揃っているか
- [ ] 各スキルについて3名以上が習得しているか
- [ ] 各役職に3名以上のメンバーがいるか
- [ ] スキル間の依存性が明確に存在するか（SkillDependencyAnalyzerで確認）

**After実装**
- [ ] モデルの適合度指標が0.7以上か（GFI, CFI等）
- [ ] パス係数が統計的に有意か（p < 0.05）
- [ ] 推薦精度（Precision@5等）が現在比10%以上向上しているか
- [ ] UI上で推薦理由が自然な日本語で表現されるか

### 参考リソース

**既存実装の活用**
- `skillnote_recommendation/core/skill_dependency_analyzer.py`: 依存性検出ロジック
- `skillnote_recommendation/graph/role_based_growth_path.py`: 役職別パス分析
- `skillnote_recommendation/utils/visualization.py`: 可視化機能

**外部ライブラリ**
- `semopy`: Python版SEM（軽量・使いやすい）
- `statsmodels`: 統計検定機能が豊富
- `pymc`: ベイズSEM（更に高度な分析）

