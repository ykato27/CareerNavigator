# 設定管理システム統合 - マイグレーションガイド

## 概要

CareerNavigatorの設定管理システムを統合し、3つの異なる設定ファイルから1つの統一されたPydantic-basedシステムに移行しました。

## 変更内容

### 旧システム（非推奨）

1. **`skillnote_recommendation/config.py`** - シンプルなdataclass
2. **`skillnote_recommendation/core/config.py`** - レガシーdict-based Config
3. **`skillnote_recommendation/core/config_v2.py`** - モダンfrozen dataclass

### 新システム（推奨）

**`skillnote_recommendation/settings.py`** - Pydantic BaseSettings

## 新システムの利点

✅ **型安全性**: Pydanticによる厳密な型チェック
✅ **環境変数サポート**: .envファイルからの自動読み込み
✅ **環境別設定**: dev/staging/prod環境の明確な分離
✅ **バリデーション**: 設定値の自動検証
✅ **Immutable**: 設定の不変性保証
✅ **統一されたAPI**: 一貫したインターフェース

## マイグレーション手順

### 1. 依存関係のインストール

```bash
pip install pydantic-settings>=2.0.0
```

### 2. コードの更新

#### Before（旧システム）

```python
# パターン1: config.py
from skillnote_recommendation.config import config
print(config.paths.DATA_DIR)

# パターン2: core/config.py
from skillnote_recommendation.core.config import Config
data_dir = Config.DATA_DIR
params = Config.MF_PARAMS

# パターン3: core/config_v2.py
from skillnote_recommendation.core.config_v2 import get_config
config = get_config(env="prod")
print(config.mf.n_components)
```

#### After（新システム）

```python
from skillnote_recommendation.settings import get_settings

# 環境変数 APP_ENV から自動取得（デフォルト: dev）
settings = get_settings()

# または明示的に環境を指定
settings = get_settings(env="prod")

# パス取得
print(settings.paths.data_dir)

# モデルパラメータ取得
print(settings.mf.n_components)
print(settings.graph.walk_length)

# 推薦パラメータ
print(settings.recommendation.category_importance_weight)
```

### 3. 環境変数の設定

#### .env ファイルの作成

```bash
# .env
APP_ENV=dev

# パス設定（オプション）
PATH__DATA_DIR=/custom/data/path
PATH__OUTPUT_DIR=/custom/output/path

# MF設定の上書き（オプション）
MF__N_COMPONENTS=30
MF__MAX_ITER=2000
MF__USE_CONFIDENCE_WEIGHTING=true

# ログレベル（オプション）
LOG__LEVEL=DEBUG
```

#### 環境別設定ファイル

```bash
# 開発環境
.env

# ステージング環境
.env.staging

# 本番環境
.env.prod
```

#### 使用例

```python
# .env.prod を使用
settings = get_settings(env="prod", env_file=".env.prod")
```

### 4. 主要な設定パスの対応表

| 旧システム | 新システム |
|-----------|-----------|
| `Config.DATA_DIR` | `settings.paths.data_dir` |
| `Config.OUTPUT_DIR` | `settings.paths.output_dir` |
| `Config.MF_PARAMS['n_components']` | `settings.mf.n_components` |
| `Config.MF_PARAMS['use_confidence_weighting']` | `settings.mf.use_confidence_weighting` |
| `Config.GRAPH_PARAMS['member_similarity_threshold']` | `settings.graph.member_similarity_threshold` |
| `Config.RECOMMENDATION_PARAMS['category_importance_weight']` | `settings.recommendation.category_importance_weight` |
| `Config.OPTUNA_PARAMS['n_trials']` | `settings.optuna.n_trials` |
| `Config.EVALUATION_PARAMS['top_k']` | `settings.evaluation.top_k` |
| `config.paths.DATA_DIR` | `settings.paths.data_dir` |
| `config.model.RANDOM_STATE` | `settings.mf.random_state` (または `settings.graph.*`) |

### 5. ヘルパーメソッドの対応

| 旧システム | 新システム |
|-----------|-----------|
| `Config.get_input_dir('members')` | `settings.get_input_dir('members')` |
| `Config.get_output_path('skill_matrix')` | `settings.get_output_path('skill_matrix')` |
| `Config.ensure_directories()` | `settings.ensure_directories()` |

## 環境別設定の例

### 開発環境

```python
settings = get_settings(env="dev")
# - LOG_LEVEL = DEBUG
# - enable_json = False (人間可読形式)
# - debug = True
```

### ステージング環境

```python
settings = get_settings(env="staging")
# - LOG_LEVEL = INFO
# - enable_json = True (JSON形式)
# - debug = False
```

### 本番環境

```python
settings = get_settings(env="prod")
# - LOG_LEVEL = INFO
# - enable_json = True (JSON形式)
# - debug = False
```

## 設定のカスタマイズ

### 環境変数による上書き

```bash
# ネストした設定は __ でつなぐ
export MF__N_COMPONENTS=50
export MF__USE_CONFIDENCE_WEIGHTING=true
export EVALUATION__TOP_K=20
export LOG__LEVEL=WARNING
```

### コードでの上書き（非推奨）

設定は基本的にimmutableなので、環境変数または.envファイルで設定してください。

## トラブルシューティング

### Q1: 旧システムと新システムを並行して使いたい

旧システムには deprecation warning が追加されていますが、当面は並行利用可能です。
ただし、将来的には旧システムは削除される予定です。

```python
# 旧システム（警告が表示される）
from skillnote_recommendation.core.config import Config

# 新システム（推奨）
from skillnote_recommendation.settings import get_settings
```

### Q2: 環境変数が効かない

1. .envファイルがプロジェクトルートにあることを確認
2. 環境変数名のプレフィックスが正しいことを確認（例: `MF__N_COMPONENTS`）
3. `force_reload=True` でキャッシュをクリア

```python
settings = get_settings(force_reload=True)
```

### Q3: 型エラーが発生する

Pydanticは厳密な型チェックを行います。設定値が期待される型と一致しているか確認してください。

```bash
# ❌ 文字列として渡すとエラー
export MF__N_COMPONENTS="30"  # Pydanticが自動変換してくれる（この場合はOK）

# ✅ 正しい型
export MF__N_COMPONENTS=30
```

### Q4: テストで設定をモックしたい

```python
from skillnote_recommendation.settings import clear_settings_cache

# テスト前にキャッシュをクリア
clear_settings_cache()

# カスタム設定でテスト
settings = get_settings(env="dev", env_file=".env.test")
```

## 移行スケジュール

| フェーズ | 期間 | 内容 |
|---------|------|------|
| Phase 1 (現在) | 1-2週間 | 新システムの導入、旧システムとの並行稼働 |
| Phase 2 | 2-4週間 | 主要コードの移行、deprecation warningの監視 |
| Phase 3 | 4-6週間 | 全コードの移行完了確認 |
| Phase 4 | 6-8週間 | 旧システムの削除 |

## サポート

質問や問題がある場合は、以下を確認してください:

1. このマイグレーションガイド
2. `skillnote_recommendation/settings.py` のdocstring
3. Pydantic Settings公式ドキュメント: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

## 参考リンク

- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Pydantic Field Types](https://docs.pydantic.dev/latest/concepts/fields/)
- [Environment Variables Best Practices](https://12factor.net/config)
