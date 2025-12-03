# Streamlit Cloud セットアップガイド

このドキュメントでは、Streamlit CloudでCareerNavigatorアプリをデプロイする方法を説明します。

## 前提条件

- GitHubアカウント
- Streamlit Cloudアカウント（GitHubアカウントでサインイン可能）
- このリポジトリのフォークまたはクローン

## デプロイ手順

### 1. Streamlit Cloudにサインイン

https://streamlit.io/cloud にアクセスし、GitHubアカウントでサインインします。

### 2. 新しいアプリをデプロイ

1. **"New app"** ボタンをクリック
2. リポジトリ、ブランチ、メインファイルを選択：
   - **Repository**: `ykato27/CareerNavigator` (または自分のフォーク)
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
3. **Advanced settings** (オプション):
   - **Python version**: `3.11` (自動検出されます)
**症状:**
```
⚠️ Optunaがインストールされていません。
```

**原因:**
- Streamlit Cloudが古いデプロイメントをキャッシュしている
- 依存関係のインストールが失敗している

**解決方法:**

#### 方法1: アプリを再起動

1. Streamlit Cloudの管理画面を開く
2. 右上のメニュー (⋮) をクリック
3. **"Reboot app"** を選択
4. アプリが再起動するまで待つ（1-3分）

#### 方法2: キャッシュをクリア

1. Streamlit Cloudの管理画面を開く
2. 右上のメニュー (⋮) をクリック
3. **"Clear cache"** を選択
4. その後、**"Reboot app"** を実行

#### 方法3: 再デプロイ

1. GitHubにダミーコミットをプッシュ（改行を追加するなど）
2. Streamlit Cloudが自動的に再デプロイを開始
3. デプロイログで `optuna` がインストールされていることを確認

#### 方法4: デプロイログを確認
**症状:**
```
Python version mismatch
```

**解決方法:**

このプロジェクトには `.python-version` ファイルが含まれており、Python 3.11を指定しています。

Streamlit Cloudは自動的にこのファイルを読み取り、指定されたバージョンを使用します。

手動で変更する場合：
1. Streamlit Cloud管理画面で **"Settings"** を開く
2. **"Python version"** を `3.11` に設定
3. **"Save"** をクリック

### データファイルがない

**症状:**
```
⚠️ まずデータを読み込んでください。
```

**原因:**
- CSVデータファイルはGitリポジトリに含まれていません（機密情報保護のため）

**解決方法:**
1. Streamlit Cloudでアプリを起動後、UIからCSVファイルをアップロード
2. または、Streamlit Cloud Secretsを使用してS3などのストレージから読み込む

## Streamlit Cloud Secrets の使用（オプション）

外部データソースやAPIキーを使用する場合、Streamlit Cloud Secretsを使用できます。

1. Streamlit Cloud管理画面で **"Settings"** を開く
2. **"Secrets"** タブを選択
3. TOML形式で秘密情報を追加：
   ```toml
   [data]
   s3_bucket = "your-bucket-name"
   s3_key = "path/to/data.csv"

   [aws]
   access_key_id = "YOUR_ACCESS_KEY"
   secret_access_key = "YOUR_SECRET_KEY"
   ```
4. コードから参照：
   ```python
   import streamlit as st

   bucket = st.secrets["data"]["s3_bucket"]
   ```

## パフォーマンス最適化

### キャッシュの活用

Streamlit Cloudでは、`@st.cache_data` と `@st.cache_resource` を活用してパフォーマンスを向上させましょう：

```python
@st.cache_data
def load_data():
    # 重い処理
    return data

@st.cache_resource
def get_model():
    # モデルの初期化
    return model
```

### リソース制限

Streamlit Cloud無料プランの制限：
- **メモリ**: 1GB
- **CPU**: 共有
- **ストレージ**: 一時的

大規模データやモデルを扱う場合は、有料プランの検討が必要です。

## サポート

問題が解決しない場合：
1. このリポジトリのIssuesを確認
2. Streamlit Cloudのドキュメントを参照: https://docs.streamlit.io/streamlit-community-cloud
3. Streamlit Communityフォーラム: https://discuss.streamlit.io/

## 関連リンク

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit App Dependencies](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
