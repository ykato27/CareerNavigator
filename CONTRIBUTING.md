# コントリビューションガイド

CareerNavigatorプロジェクトへの貢献を歓迎します！

このガイドでは、開発環境のセットアップから、コーディング規約、プルリクエストの提出までを説明します。

---

## 目次

1. [開発環境のセットアップ](#開発環境のセットアップ)
2. [ブランチ戦略](#ブランチ戦略)
3. [コーディング規約](#コーディング規約)
4. [テスト](#テスト)
5. [コミットメッセージ](#コミットメッセージ)
6. [プルリクエスト](#プルリクエスト)
7. [ドキュメント](#ドキュメント)

---

## 開発環境のセットアップ

### 必要な環境

- **Python 3.11以上** (推奨: 3.12+)
- **uv** (Pythonパッケージマネージャー)
- **Git**

### セットアップ手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/ykato27/CareerNavigator.git
cd CareerNavigator

# 2. 開発用依存関係を含めてインストール
uv sync --all-extras

# 3. pre-commit フックの設定（推奨）
uv run pre-commit install

# 4. テストを実行して動作確認
uv run pytest
```

### IDE設定

#### VSCode

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.rulers": [100]
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. `.venv/bin/python` を選択
3. Tools → Black → Enable Black formatter
4. Tools → External Tools → Flake8, mypy を追加

---

## ブランチ戦略

### ブランチ命名規則

```
<type>/<short-description>

例:
feature/add-new-recommender
fix/cold-start-error-handling
refactor/config-management
docs/update-api-reference
```

### ブランチタイプ

- `feature/`: 新機能の追加
- `fix/`: バグ修正
- `refactor/`: リファクタリング
- `docs/`: ドキュメントの更新
- `test/`: テストの追加・修正
- `chore/`: その他（依存関係更新等）

### ワークフロー

```bash
# 1. mainブランチから最新を取得
git checkout main
git pull origin main

# 2. 作業ブランチを作成
git checkout -b feature/my-new-feature

# 3. 変更を加える
# ...

# 4. コミット（詳細は後述）
git add .
git commit -m "feat: Add new feature"

# 5. プッシュ
git push origin feature/my-new-feature

# 6. プルリクエストを作成
```

---

## コーディング規約

### Python Style Guide

- **ベース**: [PEP 8](https://pep8-ja.readthedocs.io/)
- **行の長さ**: 最大100文字
- **フォーマッター**: Black
- **Linter**: Flake8
- **型チェック**: mypy (strict mode)

### コード品質チェック

開発中に以下のコマンドを実行してください：

```bash
# フォーマット
uv run black skillnote_recommendation/

# Lint
uv run flake8 skillnote_recommendation/

# 型チェック
uv run mypy skillnote_recommendation/

# 全て実行
uv run black . && uv run flake8 . && uv run mypy skillnote_recommendation/
```

### 型ヒント

**必須**: 全ての関数に型ヒントを付けてください。

```python
# Good
def recommend(
    member_code: str,
    top_n: int = 10,
    use_diversity: bool = True
) -> list[Recommendation]:
    ...

# Bad
def recommend(member_code, top_n=10, use_diversity=True):
    ...
```

### ドキュメンテーション

**必須**: 全てのpublic関数・クラスにdocstringを記述してください。

```python
def evaluate_recommendations(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    top_k: int = 10
) -> dict[str, float]:
    """
    推薦結果を評価

    Args:
        train_data: 訓練データ
        test_data: テストデータ
        top_k: 評価する推薦数

    Returns:
        評価メトリクスの辞書

    Raises:
        DataInvalidError: データが不正な場合
    """
    ...
```

### インポート順序

```python
# 1. 標準ライブラリ
import os
import sys
from typing import Optional, List

# 2. サードパーティライブラリ
import pandas as pd
import numpy as np
from pydantic import BaseModel

# 3. ローカルモジュール
from skillnote_recommendation.core.config import Config
from skillnote_recommendation.core.errors import ColdStartError
```

### 命名規則

- **クラス**: PascalCase (`MatrixFactorizationModel`)
- **関数**: snake_case (`get_recommendations`)
- **定数**: UPPER_SNAKE_CASE (`MAX_ITERATIONS`)
- **プライベート**: アンダースコアプレフィックス (`_internal_method`)

---

## テスト

### テストの実行

```bash
# 全テスト実行
uv run pytest

# カバレッジ付き
uv run pytest --cov=skillnote_recommendation

# 特定のテストのみ
uv run pytest tests/test_matrix_factorization.py

# マーカーを使った実行
uv run pytest -m "not slow"
```

### テストの作成

**場所**: `tests/test_<module_name>.py`

**命名規則**: `test_<function_name>_<scenario>`

```python
import pytest
from skillnote_recommendation.ml import MatrixFactorizationModel

class TestMatrixFactorization:
    """Matrix Factorizationモデルのテスト"""

    def test_fit_creates_correct_shapes(self, skill_matrix):
        """fit()が正しい形状の因子行列を生成することを検証"""
        model = MatrixFactorizationModel(n_components=10)
        model.fit(skill_matrix)

        assert model.W.shape == (skill_matrix.shape[0], 10)
        assert model.H.shape == (10, skill_matrix.shape[1])
        assert model.is_fitted == True

    def test_predict_raises_error_for_unknown_member(self, fitted_model):
        """未知のメンバーに対してエラーを発生させることを検証"""
        with pytest.raises(ValueError, match="学習データに存在しません"):
            fitted_model.predict("UNKNOWN_MEMBER")
```

### フィクスチャ

共通のテストデータは `tests/conftest.py` に定義：

```python
import pytest
import pandas as pd

@pytest.fixture
def skill_matrix():
    """テスト用スキルマトリクス"""
    return pd.DataFrame(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
        index=["M001", "M002", "M003"],
        columns=["C001", "C002", "C003"]
    )
```

### カバレッジ目標

- **最小**: 70%
- **推奨**: 80%以上
- **クリティカルパス**: 100%

---

## コミットメッセージ

### フォーマット

[Conventional Commits](https://www.conventionalcommits.org/ja/)に従います。

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

- `feat`: 新機能
- `fix`: バグ修正
- `refactor`: リファクタリング
- `docs`: ドキュメント
- `test`: テスト
- `chore`: その他（依存関係更新等）
- `perf`: パフォーマンス改善
- `style`: コードスタイル

### 例

```
feat(ml): Add diversity reranking with MMR algorithm

Implement Maximal Marginal Relevance (MMR) algorithm for
diversity-aware reranking of recommendations.

- Add DiversityReranker class with MMR implementation
- Support position-aware ranking
- Add caching for similarity calculations

Closes #123
```

```
fix(core): Handle cold start error in recommendation engine

When a member has no competence data, the system now raises
ColdStartError instead of generic ValueError.

Fixes #456
```

### コミットサイズ

- 小さく、論理的にまとまった変更を1コミットに
- 複数の無関係な変更を1コミットにしない
- 大きな変更は複数のコミットに分割

---

## プルリクエスト

### PR作成前のチェックリスト

- [ ] コードフォーマット（Black）を実行
- [ ] Lint（Flake8）をパス
- [ ] 型チェック（mypy）をパス
- [ ] 全テストをパス
- [ ] 新しいコードにテストを追加
- [ ] ドキュメントを更新

### PRテンプレート

```markdown
## 概要
<!-- このPRの目的を簡潔に説明 -->

## 変更内容
<!-- 主な変更点をリスト -->
- 変更1
- 変更2
- 変更3

## 関連Issue
<!-- 関連するIssueがあればリンク -->
Closes #123

## テスト
<!-- テスト方法を説明 -->
- [ ] ユニットテスト追加
- [ ] 統合テスト追加
- [ ] 手動テスト実施

## スクリーンショット
<!-- UIに変更がある場合はスクリーンショット -->

## Breaking Changes
<!-- 破壊的変更がある場合は明記 -->
なし / あり（詳細: ...）

## チェックリスト
- [ ] コードフォーマット実施（Black）
- [ ] Lint通過（Flake8）
- [ ] 型チェック通過（mypy）
- [ ] テスト通過
- [ ] ドキュメント更新
```

### レビュープロセス

1. **PR作成**: ブランチをプッシュしてPRを作成
2. **自動チェック**: CI/CDが自動的にテスト・Lintを実行
3. **コードレビュー**: レビュアーが変更を確認
4. **修正**: フィードバックに基づいて修正
5. **承認**: レビュアーが承認
6. **マージ**: main ブランチにマージ

### レビュー観点

- コードの正確性
- 可読性
- パフォーマンス
- セキュリティ
- テストカバレッジ
- ドキュメント

---

## ドキュメント

### ドキュメントの種類

- **README.md**: プロジェクト概要、使用方法
- **ARCHITECTURE.md**: アーキテクチャ設計
- **API_REFERENCE.md**: API使用方法
- **REFACTORING_GUIDE.md**: リファクタリングガイド
- **CONTRIBUTING.md**: 本ドキュメント
- **docs/*.md**: 各種ガイド

### ドキュメント更新タイミング

- 新機能追加時: API_REFERENCE.md, README.md
- アーキテクチャ変更時: ARCHITECTURE.md
- 使用方法変更時: README.md, 関連ガイド

### ドキュメント品質

- 明確で簡潔な説明
- コード例を含める
- 最新の状態に保つ

---

## FAQ

### Q: 新しい依存関係を追加したい

```bash
# 本番依存関係
uv add <package-name>

# 開発依存関係
uv add --dev <package-name>
```

### Q: テストデータの追加方法は？

`tests/conftest.py` にフィクスチャとして追加してください。

### Q: v2.0.0の新機能を使うべき？

はい。新しいコードでは v2.0.0 の新機能（構造化ロギング、Config V2等）を使用してください。
既存コードは段階的に移行してください。

### Q: 後方互換性を壊す変更は可能？

メジャーバージョンアップ時のみ可能です。事前に議論してください。

---

## サポート

質問・問題がある場合：

1. **Issue検索**: 既存のIssueを検索
2. **新しいIssue作成**: 見つからなければ新規作成
3. **ディスカッション**: GitHub Discussionsで議論

---

## ライセンス

このプロジェクトに貢献することで、あなたのコントリビューションがプロジェクトと同じライセンスの下でライセンスされることに同意したものとみなされます。
