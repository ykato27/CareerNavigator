# テスト設計ドキュメント - CareerNavigator WebUI

**バージョン**: 3.0  
**最終更新**: 2025-12-04  
**対象プロジェクト**: CareerNavigator WebUI（React + FastAPI）

---

## 目次

1. [概要](#概要)
2. [テスト戦略](#テスト戦略)
3. [テスト構成](#テスト構成)
4. [バックエンドテスト設計](#バックエンドテスト設計)
5. [テストデータ戦略](#テストデータ戦略)
6. [テスト実行環境](#テスト実行環境)
7. [カバレッジ目標と現状](#カバレッジ目標と現状)

---

## 1. 概要

### 1.1 目的

このドキュメントは、CareerNavigator WebUIの包括的なテスト設計を定義します。品質保証とリグレッション防止のため、バックエンドAPIに対する詳細なテストケースを提供します。

### 1.2 対象システム

**システム名**: CareerNavigator WebUI

**アーキテクチャ**:
- フロントエンド: React + TypeScript + Vite
- バックエンド: FastAPI + Pydantic
- コアライブラリ: skillnote_recommendation

**主要機能**:
- モデル学習API
- スキル推薦API
- 推薦重み調整API
- 従業員キャリアダッシュボード

### 1.3 現在のテスト状況（2025-12-04時点）

**Phase 2完了: バックエンドコア層 100%達成** ✅

**完了済みテスト**:
- ✅ Servicesレイヤー (96-100%カバレッジ)
- ✅ Repositoriesレイヤー (100%カバレッジ)
- ✅ Schemasレイヤー (100%カバレッジ)
- ✅ Middlewareレイヤー (100%カバレッジ)

**総テスト数**: 81+  
**課題**:
- API層のカバレッジ向上が必要
- フロントエンドテストが未実装

---

## 2. テスト戦略

### 2.1 テストレベル

| レベル | 目的 | スコープ | 優先度 |
|--------|------|----------|--------|
| **単体テスト** | 個別関数/クラスの正確性検証 | Services, Repositories, Schemas, Middleware | 最高 |
| **統合テスト** | APIエンドポイントの動作検証 | FastAPI routes | 高 |
| **E2Eテスト** | ユーザーシナリオ検証 | Web UI全体 | 中 |

### 2.2 テストツール

**バックエンド**:
- テストフレームワーク: pytest 7.0+
- 非同期テスト: pytest-asyncio
- モック/スタブ: pytest-mock, unittest.mock
- カバレッジ: pytest-cov
- 依存管理: uv

**フロントエンド（将来実装予定）**:
- テストフレームワーク: Vitest
- コンポーネントテスト: React Testing Library  
- E2Eテスト: Playwright

---

## 3. テスト構成

### 3.1 テストディレクトリ構造

```
tests/backend/
├── conftest.py                         # 共通フィクスチャ
├── test_services_training.py           # 学習サービステスト
├── test_services_recommendation.py      # 推薦サービステスト
├── test_services_weights.py            # 重み調整サービステスト
├── test_repositories_session.py        # セッションリポジトリテスト
├── test_schemas_request.py             # リクエストスキーマテスト
├── test_schemas_response.py            # レスポンススキーマテスト
├── test_middleware_error_handler.py    # エラーハンドラーテスト
├── test_middleware_logging.py          # ロギングミドルウェアテスト
├── test_api_train.py                   # 学習APIテスト
├── test_api_recommendation.py          # 推薦APIテスト
└── test_api_weights.py                 # 重み調整APIテスト
```

### 3.2 共通フィクスチャ（conftest.py）

```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def mock_session_data():
    """Mock session data for testing"""
    return {
        "session_id": "test_session",
        "model": MockRecommender(),
        "metadata": {...}
    }
```

---

## 4. バックエンドテスト設計

### 4.1 Services層テスト（100%達成）

#### TrainingService (`test_services_training.py`)

**テストケース**:
- ✅ モデル学習の成功
- ✅ カスタム重みでの学習
- ✅ セッション未登録エラー
- ✅ 不十分なデータエラー
- ✅ モデルサマリー取得
- ✅ モデル削除
- ✅ シングルトンインスタンス検証

#### RecommendationService (`test_services_recommendation.py`)

**テストケース**:
- ✅ 推薦生成の成功
- ✅ top_nパラメータの動作
- ✅ 空の結果処理
- ✅ 例外ハンドリング
- ✅ メタデータ生成

#### WeightsService (`test_services_weights.py`)

**テストケース**:
- ✅ 重み更新の成功
- ✅ モデル未登録エラー
- ✅ バリデーションエラー
- ✅ 現在の重み取得
- ✅ デフォルト重みフォールバック

### 4.2 Repositories層テスト（100%達成）

#### SessionRepository (`test_repositories_session.py`)

**テストケース**:
- ✅ セッション登録・取得
- ✅ モデル登録・取得
- ✅ セッション削除
- ✅ 存在確認
- ✅ 統計情報取得

### 4.3 Schemas層テスト（100%達成）

#### Request Schemas (`test_schemas_request.py`)

**テストケース**:
- ✅ TrainModelRequest検証
- ✅ UpdateWeightsRequest検証（合計1.0、負値チェック）
- ✅ GetRecommendationsRequest検証
- ✅ Pydanticバリデーションエラー

#### Response Schemas (`test_schemas_response.py`)

**テストケース**:
- ✅ TrainingResponse構造検証
- ✅ RecommendationsResponse構造検証
- ✅ WeightsResponse構造検証
- ✅ ErrorResponse構造検証

### 4.4 Middleware層テスト（100%達成）

#### ErrorHandler (`test_middleware_error_handler.py`)

**テストケース**:
- ✅ 成功リクエストの通過
- ✅ AppException処理
- ✅ 予期しない例外処理
- ✅ trace_id生成

#### Logging (`test_middleware_logging.py`)

**テストケース**:  
- ✅ リクエスト/レスポンスログ
- ✅ 例外時のログ
- ✅ コンテキストバインド/クリア

### 4.5 API層テスト（部分実装）

#### Train API (`test_api_train.py`)

**既存テスト**:
- モデル学習エンドポイント
- モデルサマリー取得
- モデル削除

**追加必要**:
- エラーケースの網羅的カバレッジ

#### Recommendation API (`test_api_recommendation.py`)

**既存テスト**:
- 推薦生成エンドポ イント

**追加必要**:
- エラーケースの網羅的カバレッジ

#### Weights API (`test_api_weights.py`)

**既存テスト**:
- 重み更新エンドポイント
- 重み取得エンドポイント

**追加必要**:
- エラーケースの網羅的カバレッジ

---

## 5. テストデータ戦略

### 5.1 テストデータの種類

1. **Mockデータ**: 単体テスト用（Services, Repositoriesレイヤー）
2. **Fixtureデータ**: 統合テスト用（APIレイヤー）
3. **合成データ**: E2Eテスト用

### 5.2 モック戦略

```python
from unittest.mock import Mock, AsyncMock

# 非同期関数のモック
mock_recommender = Mock()
mock_recommender.recommend = Mock(return_value=[...])

# Exceptionのモック
mock_recommender.recommend.side_effect = Exception("Test error")
```

---

## 6. テスト実行環境

### 6.1 ローカル環境

```bash
# 全テスト実行
uv run pytest tests/backend/

# カバレッジ付き実行
uv run pytest tests/backend/ --cov=backend --cov-report=term-missing

# HTMLレポート生成
uv run pytest tests/backend/ --cov=backend --cov-report=html

# 特定のモジュールのみ
uv run pytest tests/backend/test_services_training.py -v
```

### 6.2 カバレッジ設定（pyproject.toml）

```toml
[tool.coverage.report]
fail_under = 90
show_missing = true
```

---

## 7. カバレッジ目標と現状

### 7.1 現在のカバレッジ（Phase 2完了時点）

| レイヤー | カバレッジ | 状態 |
|---------|-----------|------|
| `backend/services/` | 96-100% | ✅ 完了 |
| `backend/repositories/` | 100% | ✅ 完了 |
| `backend/schemas/` | 100% | ✅ 完了 |
| `backend/middleware/` | 100% | ✅ 完了 |
| `backend/api/` | 40-60% | ⏳ 改善必要 |
| `backend/utils/` | 50-70% | ⏳ 改善必要 |

**全体カバレッジ**: ~70%（Phase 2完了時）  
**最終目標**: 90%以上

### 7.2 次フェーズ（Phase 3以降）

**Phase 3: API層カバレッジ向上**
- ✅ Services/Repositories/Schemasのテストインフラ確立
- ⏳ APIエラーケースの網羅的テスト追加
- ⏳ utils層の単体テスト追加

**Phase 4: フロントエンドテスト（将来）**
- ⏳ Vitestセットアップ
- ⏳ Reactコンポーネントテスト
- ⏳ Playwrigh E2Eテスト

---

## まとめ

**Phase 2完了**: バックエンドのコア層（Services, Repositories, Schemas, Middleware）で100%カバレッジを達成。81+のテストが実装され、プロジェクトの品質基盤が確立されました。

**次のステップ**:
1. API層のエラーケーステストを追加してカバレッジ90%を目指す
2. utils層の単体テストを追加
3. GitHub Actionsでの継続的テスト実行
4. フロントエンドテストフレームワークのセットアップ（長期）
