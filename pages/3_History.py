"""
キャリア推薦システム - 推薦履歴ページ

このページでは、過去の推薦結果の履歴を表示します。
ログイン済みユーザーのみ利用可能です。

主な機能:
- 推薦履歴の一覧表示
- 履歴の詳細表示
- 履歴の検索とフィルタリング
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from skillnote_recommendation.core.persistence.streamlit_integration import StreamlitPersistenceManager


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="キャリア推薦システム - 履歴",
    page_icon="📜",
    layout="wide"
)


# =========================================================
# 永続化マネージャーの初期化
# =========================================================
@st.cache_resource
def get_persistence_manager():
    """永続化マネージャーのシングルトンインスタンスを取得"""
    return StreamlitPersistenceManager()


persistence_manager = get_persistence_manager()
persistence_manager.initialize_session()

# ユーザーログインUI
persistence_manager.render_user_login()


# =========================================================
# ヘッダー
# =========================================================

st.title("📜 推薦履歴")
st.markdown("**過去の推薦結果を確認できます。**")


# =========================================================
# ログイン確認
# =========================================================

current_user = persistence_manager.get_current_user()
if not current_user:
    st.warning("⚠️ 履歴を表示するにはログインが必要です。")
    st.info("👉 サイドバーからログインしてください。")
    st.stop()


# =========================================================
# 履歴読み込み
# =========================================================

st.subheader(f"👤 {current_user.username} の推薦履歴")

# フィルタリングオプション
col1, col2 = st.columns(2)

with col1:
    limit = st.slider(
        "表示件数",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
        help="表示する履歴の最大件数"
    )

with col2:
    filter_member = st.text_input(
        "メンバーコードでフィルタ（任意）",
        help="特定のメンバーの履歴のみを表示"
    )

# 履歴を読み込み
if filter_member:
    history = persistence_manager.load_user_history(
        limit=limit,
        member_code=filter_member
    )
else:
    history = persistence_manager.load_user_history(limit=limit)


# =========================================================
# 履歴表示
# =========================================================

if not history:
    st.info("まだ推薦履歴がありません。")
    st.markdown("推論ページで推薦を実行すると、履歴が保存されます。")
    st.stop()

st.success(f"✅ {len(history)}件の履歴が見つかりました")

# 統計情報
st.markdown("---")
st.subheader("📊 統計情報")

col1, col2, col3, col4 = st.columns(4)

# 推薦方法の集計
methods = [h.method for h in history]
method_counts = pd.Series(methods).value_counts()

# メンバー数の集計
unique_members = len(set(h.member_code for h in history))

# 平均実行時間
avg_time = sum(h.execution_time for h in history if h.execution_time) / len(history)

# 総推薦数
total_recs = sum(len(h.recommendations) for h in history)

with col1:
    st.metric("履歴件数", len(history))

with col2:
    st.metric("対象メンバー数", unique_members)

with col3:
    st.metric("平均実行時間", f"{avg_time:.2f}秒")

with col4:
    st.metric("総推薦数", total_recs)

# 推薦方法の分布
with st.expander("📈 推薦方法の分布"):
    st.bar_chart(method_counts)


# =========================================================
# 履歴一覧
# =========================================================

st.markdown("---")
st.subheader("📋 履歴一覧")

# 履歴を表形式で表示
for i, record in enumerate(history, 1):
    with st.expander(
        f"#{i} | {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{record.member_name} ({record.member_code}) | "
        f"{record.method} | "
        f"{len(record.recommendations)}件"
    ):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📝 基本情報")
            st.write(f"**履歴ID**: `{record.history_id}`")
            st.write(f"**メンバーコード**: {record.member_code}")
            st.write(f"**メンバー名**: {record.member_name}")
            st.write(f"**推薦方法**: {record.method}")
            st.write(f"**実行日時**: {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if record.execution_time:
                st.write(f"**実行時間**: {record.execution_time:.3f}秒")
            st.write(f"**推薦数**: {len(record.recommendations)}件")

        with col2:
            st.markdown("### ⚙️ パラメータ")
            if record.parameters:
                params_df = pd.DataFrame([
                    {"パラメータ": k, "値": str(v)}
                    for k, v in record.parameters.items()
                    if v is not None
                ])
                if not params_df.empty:
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                else:
                    st.info("パラメータなし")
            else:
                st.info("パラメータなし")

        # 推薦結果の詳細
        st.markdown("### 🎯 推薦結果")

        if record.recommendations:
            recs_df = pd.DataFrame(record.recommendations)

            # カラム名を日本語に変換
            column_mapping = {
                "competence_code": "力量コード",
                "competence_name": "力量名",
                "competence_type": "力量タイプ",
                "category": "カテゴリー",
                "priority_score": "優先度スコア",
                "reason": "推薦理由"
            }

            # 存在するカラムのみマッピング
            recs_df = recs_df.rename(columns={
                k: v for k, v in column_mapping.items() if k in recs_df.columns
            })

            # データフレームを表示
            st.dataframe(
                recs_df,
                use_container_width=True,
                hide_index=True
            )

            # CSVダウンロードボタン
            csv = recs_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 この履歴をCSVでダウンロード",
                data=csv,
                file_name=f"recommendation_history_{record.history_id}.csv",
                mime="text/csv",
                key=f"download_{record.history_id}"
            )
        else:
            st.info("推薦結果がありません")


# =========================================================
# 保存済みモデルの表示
# =========================================================

st.markdown("---")
st.subheader("💾 保存済みモデル")

models = persistence_manager.list_saved_models()

if not models:
    st.info("保存済みモデルがありません")
else:
    st.success(f"✅ {len(models)}件の保存済みモデルがあります")

    for model in models:
        with st.expander(
            f"{model['model_type'].upper()} モデル | "
            f"{model['created_at'][:19]} | "
            f"再構成誤差: {model.get('metrics', {}).get('reconstruction_error', 'N/A')}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📝 モデル情報")
                st.write(f"**モデルID**: `{model['model_id'][:16]}...`")
                st.write(f"**モデルタイプ**: {model['model_type']}")
                st.write(f"**作成日時**: {model['created_at'][:19]}")
                if model.get('description'):
                    st.write(f"**説明**: {model['description']}")

            with col2:
                st.markdown("### ⚙️ パラメータ")
                if model.get('parameters'):
                    params_df = pd.DataFrame([
                        {"パラメータ": k, "値": str(v)}
                        for k, v in model['parameters'].items()
                    ])
                    st.dataframe(params_df, use_container_width=True, hide_index=True)

            if model.get('metrics'):
                st.markdown("### 📊 メトリクス")
                metrics_df = pd.DataFrame([
                    {"メトリクス": k, "値": f"{v:.6f}" if isinstance(v, float) else str(v)}
                    for k, v in model['metrics'].items()
                ])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# =========================================================
# フッター
# =========================================================

st.markdown("---")
st.info("💡 推薦履歴とモデルはデータベースに保存されており、ブラウザを閉じても保持されます。")
