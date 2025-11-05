# データ永続化とセッション管理

## 概要

CareerNavigatorシステムにデータ永続化とセッション管理機能が追加されました。この機能により、以下が可能になります：

- **ユーザー管理**: 個別のユーザーアカウントでログイン
- **推薦履歴の保存**: 過去の推薦結果を保存・参照
- **モデルの永続化**: 学習済みモデルを保存して再利用
- **セッション管理**: ブラウザを閉じてもデータが保持

## アーキテクチャ

### レイヤー構成

```
┌─────────────────────────────────────────┐
│   Streamlit Integration Layer           │
│   - StreamlitPersistenceManager         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Application Layer                      │
│   - UserRepository                       │
│   - RecommendationHistoryRepository      │
│   - ModelRepository                      │
│   - SessionManager                       │
│   - ModelStorage                         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Data Layer                             │
│   - DatabaseManager (SQLite)             │
│   - Models (User, History, etc.)         │
└─────────────────────────────────────────┘
```

### データモデル

#### User（ユーザー）
- `user_id`: ユーザーID（UUID）
- `username`: ユーザー名
- `email`: メールアドレス（任意）
- `created_at`: 作成日時
- `last_login`: 最終ログイン日時
- `settings`: ユーザー設定（JSON）

#### RecommendationHistory（推薦履歴）
- `history_id`: 履歴ID（UUID）
- `user_id`: ユーザーID
- `member_code`: 対象メンバーコード
- `member_name`: 対象メンバー名
- `method`: 推薦方法
- `timestamp`: 実行日時
- `recommendations`: 推薦結果（JSON）
- `reference_persons`: 参考人物（JSON）
- `parameters`: パラメータ（JSON）
- `execution_time`: 実行時間（秒）

#### ModelMetadata（モデルメタデータ）
- `model_id`: モデルID
- `user_id`: ユーザーID
- `model_type`: モデルタイプ（nmf, graph, hybrid）
- `created_at`: 作成日時
- `parameters`: モデルパラメータ（JSON）
- `metrics`: 評価メトリクス（JSON）
- `file_path`: モデルファイルパス
- `data_hash`: 学習データハッシュ
- `description`: 説明

#### UserSession（ユーザーセッション）
- `session_id`: セッションID（UUID）
- `user_id`: ユーザーID
- `created_at`: 作成日時
- `last_active`: 最終アクティブ日時
- `data_loaded`: データ読み込み済みフラグ
- `model_trained`: モデル学習済みフラグ
- `current_model_id`: 現在のモデルID
- `state_data`: 状態データ（JSON）

## 使用方法

### 1. ユーザーログイン

アプリケーションのサイドバーにログインUIが表示されます：

1. ユーザー名を入力
2. メールアドレスを入力（任意）
3. 「ログイン」ボタンをクリック

- 既存のユーザー名の場合はログイン
- 新しいユーザー名の場合は自動的にアカウント作成

### 2. モデルの保存

モデル学習ページ（`1_Model_Training.py`）で：

1. データを読み込む
2. モデル学習オプションを選択
3. 「MLモデル学習を実行」をクリック

**ログイン済みの場合**:
- モデルは自動的にデータベースに保存されます
- モデルIDが表示されます
- 次回は学習済みモデルを再利用できます

**未ログインの場合**:
- モデルは現在のセッションのみで使用可能
- ブラウザを閉じるとモデルは失われます

### 3. 推薦履歴の保存

推論ページ（`2_Inference.py`）で：

1. メンバーを選択
2. 推薦方法とパラメータを設定
3. 「推薦を実行」をクリック

**ログイン済みの場合**:
- 推薦結果が自動的に履歴として保存されます
- 実行日時、パラメータ、推薦結果が記録されます

### 4. 履歴の確認

履歴ページ（`3_History.py`）で：

1. ログイン状態で履歴ページにアクセス
2. 過去の推薦履歴が一覧表示されます
3. フィルタリング・検索が可能
4. 各履歴の詳細を確認できます
5. CSVでダウンロード可能

**表示内容**:
- 基本情報（日時、メンバー、推薦方法）
- 実行パラメータ
- 推薦結果の詳細
- 統計情報（推薦方法の分布、平均実行時間など）

### 5. 保存済みモデルの確認

履歴ページで保存済みモデルも確認できます：

- モデルタイプ
- 作成日時
- パラメータ
- 評価メトリクス
- 説明

## データベース管理

### データベースファイル

デフォルトでは `career_navigator.db` という名前のSQLiteデータベースファイルが作成されます。

### バックアップ

データベースファイルをコピーすることで簡単にバックアップできます：

```bash
cp career_navigator.db career_navigator_backup_$(date +%Y%m%d).db
```

### データのクリーンアップ

古いセッションデータをクリーンアップする場合：

```python
from skillnote_recommendation.core.persistence import SessionManager, DatabaseManager

db = DatabaseManager()
db.initialize_schema()
session_manager = SessionManager(db)

# 30日以上古いセッションを削除
deleted_count = session_manager.cleanup_old_sessions(days=30)
print(f"Deleted {deleted_count} old sessions")
```

## プログラマティックな使用

### ユーザー作成

```python
from skillnote_recommendation.core.persistence import DatabaseManager, UserRepository

db = DatabaseManager()
db.initialize_schema()
user_repo = UserRepository(db)

# 新規ユーザー作成
user = user_repo.create_user(
    username="john_doe",
    email="john@example.com",
    settings={"theme": "dark"}
)

print(f"Created user: {user.username} (ID: {user.user_id})")
```

### 推薦履歴の保存

```python
from skillnote_recommendation.core.persistence import (
    DatabaseManager,
    RecommendationHistoryRepository,
    RecommendationHistory
)

db = DatabaseManager()
db.initialize_schema()
history_repo = RecommendationHistoryRepository(db)

# 履歴作成
history = RecommendationHistory(
    user_id="user_id_here",
    member_code="M001",
    member_name="山田太郎",
    method="NMF推薦",
    recommendations=[
        {
            "competence_code": "C001",
            "competence_name": "Python",
            "priority_score": 0.95
        }
    ],
    parameters={"top_n": 10},
    execution_time=1.5
)

saved_history = history_repo.create_history(history)
```

### モデルの保存と読み込み

```python
from skillnote_recommendation.core.persistence import (
    DatabaseManager,
    ModelStorage
)

db = DatabaseManager()
db.initialize_schema()
model_storage = ModelStorage(db)

# モデルを保存
metadata = model_storage.save_model(
    model=ml_recommender,
    user_id="user_id_here",
    model_type="nmf",
    parameters={"n_components": 10},
    metrics={"reconstruction_error": 0.05},
    description="NMF model with 10 components"
)

print(f"Saved model: {metadata.model_id}")

# モデルを読み込み
loaded_model, metadata = model_storage.load_latest_model(
    user_id="user_id_here",
    model_type="nmf"
)

if loaded_model:
    print(f"Loaded model: {metadata.model_id}")
```

## セキュリティとプライバシー

### データの保護

- データベースファイルはローカルに保存されます
- SQLインジェクション対策：パラメータ化クエリを使用
- ユーザー認証：現在はシンプルなユーザー名ベース

### 本番環境での推奨事項

1. **パスワード認証の追加**
   - 現在はユーザー名のみで認証
   - 本番環境ではパスワードハッシュの実装を推奨

2. **データベースの暗号化**
   - SQLCipherなどを使用したデータベース暗号化を検討

3. **アクセス制御**
   - データベースファイルのパーミッション設定
   - 適切なユーザー権限管理

4. **定期的なバックアップ**
   - 自動バックアップスクリプトの設定
   - バックアップの定期的なテスト

## トラブルシューティング

### データベースエラー

**問題**: `database is locked` エラー

**解決策**:
- 同時に複数のStreamlitインスタンスを実行していないか確認
- データベースファイルへの書き込み権限を確認

### モデルの読み込みエラー

**問題**: モデルファイルが見つからない

**解決策**:
- `models/` ディレクトリが存在するか確認
- モデルファイルのパスが正しいか確認
- ディスク容量を確認

### 履歴が表示されない

**問題**: 履歴ページで履歴が表示されない

**解決策**:
- ログインしているか確認
- 推薦を実行したか確認
- データベースファイルが破損していないか確認

## パフォーマンス最適化

### インデックス

データベースには以下のインデックスが作成されています：

- `idx_history_user_id`: ユーザーIDでの履歴検索
- `idx_history_timestamp`: タイムスタンプでのソート
- `idx_model_user_id`: ユーザーIDでのモデル検索
- `idx_sessions_user_id`: ユーザーIDでのセッション検索

### クエリ最適化

大量のデータがある場合：

1. **ページネーション**: `limit` と `offset` を使用
2. **フィルタリング**: 必要なデータのみを取得
3. **定期的なクリーンアップ**: 古いデータを削除

## 今後の拡張

### 計画中の機能

1. **パスワード認証**
   - ハッシュ化されたパスワードによる認証

2. **ユーザーロール**
   - 管理者、一般ユーザーなどの権限管理

3. **データエクスポート**
   - 全履歴のエクスポート機能
   - データのバックアップとリストア

4. **分析ダッシュボード**
   - 推薦履歴の統計分析
   - トレンド分析

5. **クラウドストレージ対応**
   - PostgreSQL, MySQL対応
   - クラウドストレージへのモデル保存

## 参考資料

- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)

## サポート

問題が発生した場合は、以下を確認してください：

1. ログファイル（存在する場合）
2. データベースファイルの存在とアクセス権限
3. Pythonのバージョン（3.8以上推奨）
4. 必要なパッケージのインストール

詳細な技術サポートが必要な場合は、開発チームまでお問い合わせください。
