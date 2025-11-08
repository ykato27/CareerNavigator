# Claude Development Rules for CareerNavigator

このドキュメントは、CareerNavigatorプロジェクトをClaudeで開発する際の規則と指針です。

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
1. **型アノテーション**: すべての関数に型情報があるか
2. **テスト**: 新機能に対するテストがあるか
3. **ドキュメント**: 説明・使い方が明記されているか
4. **リファクタリング**: 既存コードで対応できないか
5. **パフォーマンス**: 計算量に問題がないか

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

