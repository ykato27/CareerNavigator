"""
CareerNavigator - モデル学習と分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from skillnote_recommendation.ml.ml_recommender import MLRecommender
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header
)
from skillnote_recommendation.ml.optuna_visualization_helper import (
    generate_optuna_visualizations,
    get_best_trials_summary,
    get_pruned_trials_count,
    plot_training_history,
)


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - モデル学習",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply rich UI styles
apply_rich_ui_styles()

# リッチなヘッダー
render_gradient_header(
    title="🧭 CareerNavigator",
    icon="🤖",
    description="モデル学習と分析 - AIモデルを学習し、学習結果を分析します"
)


# =========================================================
# データ読み込みチェック
# =========================================================
if not st.session_state.get("data_loaded", False):
    st.warning("⚠️ まずデータを読み込んでください。")
    st.info("👉 サイドバーから「データ読み込み」ページに戻ってCSVファイルをアップロードしてください。")
    st.stop()


# =========================================================
# 補助関数
# =========================================================
def build_ml_recommender(
    transformed_data: dict,
    use_preprocessing: bool = True,
    use_tuning: bool = False,
    tuning_n_trials: int = None,
    tuning_timeout: int = None,
    tuning_search_space: dict = None,
    tuning_sampler: str = None,
    tuning_random_state: int = None,
    tuning_progress_callback = None
) -> MLRecommender:
    """
    MLRecommenderを学習済みの状態で作成する

    Args:
        transformed_data: 変換済みデータ
        use_preprocessing: データ前処理を使用するか
        use_tuning: ハイパーパラメータチューニングを使用するか
        tuning_n_trials: チューニング試行回数
        tuning_timeout: チューニングタイムアウト
        tuning_search_space: チューニング探索空間
        tuning_sampler: チューニングサンプラー
        tuning_random_state: チューニングの乱数シード
        tuning_progress_callback: 進捗コールバック
    """
    # データの検証
    if transformed_data is None:
        raise ValueError("transformed_data が None です。データが正しく読み込まれていません。")

    if not isinstance(transformed_data, dict):
        raise TypeError(f"transformed_data は dict 型である必要があります。実際の型: {type(transformed_data).__name__}")

    # 必要なキーの存在確認
    required_keys = ["member_competence", "competence_master", "members_clean"]
    missing_keys = [key for key in required_keys if key not in transformed_data]
    if missing_keys:
        raise KeyError(f"必要なキーが不足しています: {', '.join(missing_keys)}。利用可能なキー: {', '.join(transformed_data.keys())}")

    recommender = MLRecommender.build(
        member_competence=transformed_data["member_competence"],
        competence_master=transformed_data["competence_master"],
        member_master=transformed_data["members_clean"],
        use_preprocessing=use_preprocessing,
        use_tuning=use_tuning,
        tuning_n_trials=tuning_n_trials,
        tuning_timeout=tuning_timeout,
        tuning_search_space=tuning_search_space,
        tuning_sampler=tuning_sampler,
        tuning_random_state=tuning_random_state,
        tuning_progress_callback=tuning_progress_callback
    )
    return recommender


# =========================================================
# モデル学習
# =========================================================
st.subheader("🎓 MLモデル学習")

if st.session_state.get("model_trained", False):
    st.success("✅ MLモデルは既に学習済みです。")

    # デバッグ情報を表示（学習後も保持）
    if st.session_state.get("show_debug_info", False) and st.session_state.get("debug_messages"):
        with st.expander("🔍 デバッグ情報（前回の学習）", expanded=False):
            st.code("\n".join(st.session_state.debug_messages))

            # デバッグ情報をクリアするボタン
            if st.button("🗑️ デバッグ情報をクリア"):
                st.session_state.show_debug_info = False
                st.rerun()

    if st.button("🔄 モデルを再学習する"):
        st.session_state.model_trained = False
        st.session_state.ml_recommender = None
        st.session_state.show_debug_info = False
        st.rerun()
else:
    st.info("📚 NMF（非負値行列分解）を使用して、メンバーの力量習得パターンを学習します。")

    # 変数の初期化
    sampler_choice = "tpe"
    n_trials = 50
    random_state = 42
    custom_search_space = None

    # 学習オプション
    with st.expander("⚙️ 学習オプション", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            use_preprocessing = st.checkbox(
                "データ前処理を使用",
                value=True,
                help="外れ値除去と正規化を行います。再構成誤差の改善に効果的です。"
            )

        with col2:
            # Optunaが利用可能かチェック
            try:
                import optuna
                optuna_available = True
            except ImportError:
                optuna_available = False

            use_tuning = st.checkbox(
                "ハイパーパラメータチューニング (Optuna)",
                value=False,
                help="ベイズ最適化でハイパーパラメータを自動調整します。時間がかかりますが、最良のモデルを構築できます。",
                disabled=not optuna_available
            )

            if not optuna_available:
                st.error("⚠️ Optunaがインストールされていません。`uv pip install --system optuna` を実行してください。")

        if use_preprocessing:
            st.markdown("""
            **データ前処理の内容:**
            - 外れ値除去: 力量数が極端に少ないメンバー/保有者が少ない力量を除外
            - 正規化: Min-Maxスケーリング（0-1範囲に正規化）
            """)

        if use_tuning and optuna_available:
            st.markdown("---")
            st.markdown("### ⚙️ チューニング詳細設定")

            # サンプラー選択
            sampler_col1, sampler_col2, sampler_col3 = st.columns(3)
            with sampler_col1:
                sampler_choice = st.selectbox(
                    "探索方法（サンプラー）",
                    options=["tpe", "random", "cmaes"],
                    format_func=lambda x: {
                        "tpe": "TPE (Tree-structured Parzen Estimator) - 推奨",
                        "random": "ランダムサーチ",
                        "cmaes": "CMA-ES (進化戦略)"
                    }[x],
                    help="TPE: ベイズ最適化で効率的に探索\nランダム: ランダムに探索\nCMA-ES: 進化戦略による最適化"
                )

            with sampler_col2:
                n_trials = st.number_input(
                    "試行回数",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="探索する組み合わせの数。多いほど良い解が見つかる可能性が高まりますが、時間がかかります。"
                )

            with sampler_col3:
                random_state = st.number_input(
                    "乱数シード（Random State）",
                    min_value=0,
                    max_value=2147483647,
                    value=42,
                    step=1,
                    help="乱数シードを固定することで、同じ探索過程を再現できます。実験の再現性が必要な場合に使用します。"
                )

            # 探索範囲のデフォルト値を設定
            n_comp_min, n_comp_max = 10, 30
            alpha_w_min, alpha_w_max = 0.001, 0.5
            alpha_h_min, alpha_h_max = 0.001, 0.5
            l1_min, l1_max = 0.0, 1.0

            # 探索範囲の設定
            with st.expander("🔍 探索範囲の詳細設定", expanded=False):
                st.markdown("各パラメータの探索範囲を設定します。デフォルト値から変更する場合のみ調整してください。")

                range_col1, range_col2 = st.columns(2)

                with range_col1:
                    st.markdown("**潜在因子数 (n_components)**")
                    n_comp_min = st.number_input("最小値", min_value=5, max_value=50, value=n_comp_min, key="n_comp_min")
                    n_comp_max = st.number_input("最大値", min_value=5, max_value=50, value=n_comp_max, key="n_comp_max")

                    st.markdown("**正則化係数 W (alpha_W)**")
                    alpha_w_min = st.number_input("最小値", min_value=0.0001, max_value=1.0, value=alpha_w_min, format="%.4f", key="alpha_w_min")
                    alpha_w_max = st.number_input("最大値", min_value=0.0001, max_value=1.0, value=alpha_w_max, format="%.4f", key="alpha_w_max")

                with range_col2:
                    st.markdown("**正則化係数 H (alpha_H)**")
                    alpha_h_min = st.number_input("最小値", min_value=0.0001, max_value=1.0, value=alpha_h_min, format="%.4f", key="alpha_h_min")
                    alpha_h_max = st.number_input("最大値", min_value=0.0001, max_value=1.0, value=alpha_h_max, format="%.4f", key="alpha_h_max")

                    st.markdown("**L1比率 (l1_ratio)**")
                    l1_min = st.number_input("最小値", min_value=0.0, max_value=1.0, value=l1_min, format="%.2f", key="l1_min")
                    l1_max = st.number_input("最大値", min_value=0.0, max_value=1.0, value=l1_max, format="%.2f", key="l1_max")

            # 探索空間を構築（expanderの外で）
            custom_search_space = {
                'n_components': (int(n_comp_min), int(n_comp_max)),
                'alpha_W': (float(alpha_w_min), float(alpha_w_max)),
                'alpha_H': (float(alpha_h_min), float(alpha_h_max)),
                'l1_ratio': (float(l1_min), float(l1_max))
            }

            st.info(f"""
            **選択した設定:**
            - 探索方法: {sampler_choice.upper()}
            - 試行回数: {int(n_trials)}回
            - 推定時間: {int(n_trials) * 0.1:.1f}〜{int(n_trials) * 0.2:.1f}分
            """)
            st.warning("⏱️ チューニングには時間がかかる場合があります。")

    # 学習実行ボタン
    button_label = "🚀 MLモデル学習を実行（チューニングあり）" if use_tuning else "🚀 MLモデル学習を実行"

    if st.button(button_label, type="primary"):
        # Optunaチェック（チューニング有効時）
        if use_tuning:
            try:
                import optuna
            except ImportError:
                st.error("❌ Optunaがインストールされていません。ハイパーパラメータチューニングを実行できません。")
                st.info("💡 以下のコマンドでインストールしてください:\n```bash\nuv pip install --system optuna\n```")
                st.stop()

        # デバッグ情報をsession_stateに初期化
        st.session_state.debug_messages = []
        st.session_state.show_debug_info = True

        # デバッグ情報専用のコンテナ
        debug_container = st.container()

        with debug_container:
            st.markdown("### 🔍 デバッグ情報")
            debug_info = st.empty()

            # 初期設定を表示
            debug_messages = []
            debug_messages.append(f"✅ データ読み込み完了")
            debug_messages.append(f"✅ use_tuning={use_tuning}")
            debug_messages.append(f"✅ use_preprocessing={use_preprocessing}")

            if use_tuning:
                debug_messages.append(f"✅ sampler_choice={sampler_choice}")
                debug_messages.append(f"✅ n_trials={int(n_trials)} (型: {type(n_trials)})")
                debug_messages.append(f"✅ random_state={int(random_state)} (型: {type(random_state)})")
                debug_messages.append(f"✅ custom_search_space={custom_search_space}")

            debug_info.code("\n".join(debug_messages))
            st.session_state.debug_messages = debug_messages.copy()

        # リアルタイム可視化用のプレースホルダー
        progress_placeholder = st.empty()
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # チューニング進捗を保存するためのリスト
        trial_history = []
        callback_counter = [0]  # コールバックが呼ばれた回数

        def progress_callback(trial, study):
            """チューニングの進捗をリアルタイムで表示"""
            callback_counter[0] += 1

            trial_history.append({
                'trial': trial.number,
                'value': trial.value,
                'best_value': study.best_value
            })

            # プログレスバーを更新
            progress_pct = (trial.number + 1) / int(n_trials) if use_tuning else 1.0
            progress_placeholder.progress(
                progress_pct,
                text=f"Trial {trial.number + 1}/{int(n_trials) if use_tuning else 1} - 現在の誤差: {trial.value:.6f} - 最良: {study.best_value:.6f}"
            )

            # グラフを更新（毎回更新）
            if True:  # リアルタイム更新
                import pandas as pd
                import plotly.graph_objects as go

                df_history = pd.DataFrame(trial_history)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_history['trial'],
                    y=df_history['value'],
                    mode='markers',
                    name='各試行',
                    marker=dict(size=8, opacity=0.6, color='lightblue')
                ))
                fig.add_trace(go.Scatter(
                    x=df_history['trial'],
                    y=df_history['best_value'],
                    mode='lines',
                    name='最良値の推移',
                    line=dict(color='red', width=2)
                ))
                fig.update_layout(
                    title='ハイパーパラメータ最適化の進捗',
                    xaxis_title='Trial',
                    yaxis_title='再構成誤差',
                    height=400
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # メトリクスを表示
                col1, col2, col3 = metrics_placeholder.columns(3)
                with col1:
                    st.metric("現在の Trial", f"{trial.number + 1}/{int(n_trials) if use_tuning else 1}")
                with col2:
                    st.metric("現在の誤差", f"{trial.value:.6f}")
                with col3:
                    st.metric("最良誤差", f"{study.best_value:.6f}")

        with st.spinner("MLモデルを学習中..." if not use_tuning else "ハイパーパラメータチューニング中..."):
            try:
                # 追加のデバッグ情報
                if use_tuning:
                    debug_messages.append(f"⏳ チューニング開始...")
                    debug_messages.append(f"   - n_trials={int(n_trials)}")
                    debug_messages.append(f"   - sampler={sampler_choice}")
                    debug_messages.append(f"   - callback設定={progress_callback is not None}")

                    # データの検証
                    if "member_competence" in st.session_state.transformed_data:
                        mc = st.session_state.transformed_data["member_competence"]
                        debug_messages.append(f"   - member_competence shape: {mc.shape}")
                        debug_messages.append(f"   - member_competence 列: {list(mc.columns)}")

                    debug_info.code("\n".join(debug_messages))
                    st.session_state.debug_messages = debug_messages.copy()

                # print()の出力をキャプチャ
                import sys
                from io import StringIO

                stdout_capture = StringIO()
                stderr_capture = StringIO()
                old_stdout = sys.stdout
                old_stderr = sys.stderr

                try:
                    # stdoutとstderrをキャプチャ
                    sys.stdout = stdout_capture
                    sys.stderr = stderr_capture

                    ml_recommender = build_ml_recommender(
                        st.session_state.transformed_data,
                        use_preprocessing=use_preprocessing,
                        use_tuning=use_tuning,
                        tuning_n_trials=int(n_trials) if use_tuning else None,
                        tuning_timeout=None,
                        tuning_search_space=custom_search_space if use_tuning else None,
                        tuning_sampler=sampler_choice if use_tuning else None,
                        tuning_random_state=int(random_state) if use_tuning else None,
                        tuning_progress_callback=progress_callback if use_tuning else None
                    )
                finally:
                    # stdoutとstderrを復元
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

                    # キャプチャした出力を取得
                    captured_stdout = stdout_capture.getvalue()
                    captured_stderr = stderr_capture.getvalue()

                    # デバッグメッセージに追加
                    if captured_stdout:
                        debug_messages.append(f"\n--- 標準出力 (stdout) ---")
                        debug_messages.append(captured_stdout)
                    if captured_stderr:
                        debug_messages.append(f"\n--- エラー出力 (stderr) ---")
                        debug_messages.append(captured_stderr)

                    st.session_state.debug_messages = debug_messages.copy()

                # チューニング完了のログ
                if use_tuning:
                    debug_messages.append(f"\n✅ チューニング完了")
                    debug_messages.append(f"   - 実行された試行数: {len(trial_history)}")
                    debug_messages.append(f"   - コールバック呼び出し回数: {callback_counter[0]}")
                    if ml_recommender.tuning_results:
                        debug_messages.append(f"   - 最良パラメータ: {ml_recommender.tuning_results['best_params']}")
                        debug_messages.append(f"   - 最小誤差: {ml_recommender.tuning_results['best_value']:.6f}")
                        if hasattr(ml_recommender.tuning_results.get('tuner'), 'study'):
                            study = ml_recommender.tuning_results['tuner'].study
                            debug_messages.append(f"   - Studyの試行数: {len(study.trials)}")
                    else:
                        debug_messages.append(f"   ⚠️ tuning_resultsがNone")
                        debug_messages.append(f"   ⚠️ これは、Optunaのstudy.optimize()が試行を実行しなかったことを意味します")
                        debug_messages.append(f"   ⚠️ 上記の標準出力/エラー出力を確認してください")
                    debug_info.code("\n".join(debug_messages))
                    # session_stateに最終結果を保存
                    st.session_state.debug_messages = debug_messages.copy()

                # プレースホルダーをクリア
                progress_placeholder.empty()
                chart_placeholder.empty()
                metrics_placeholder.empty()
                st.session_state.ml_recommender = ml_recommender
                st.session_state.model_trained = True

                # モデルの保存（persistence_managerが利用可能な場合）
                if 'persistence_manager' in globals():
                    current_user = persistence_manager.get_current_user()
                    if current_user:
                        with st.spinner("モデルを保存中..."):
                            try:
                                # モデルのパラメータとメトリクスを取得
                                mf_model = ml_recommender.mf_model
                                parameters = {
                                    "n_components": mf_model.n_components,
                                    "use_preprocessing": use_preprocessing,
                                    "use_tuning": use_tuning,
                                }
                                metrics = {
                                    "reconstruction_error": mf_model.get_reconstruction_error(),
                                }

                                # モデルを保存
                                model_id = persistence_manager.save_trained_model(
                                    model=ml_recommender,
                                    model_type="nmf",
                                    parameters=parameters,
                                    metrics=metrics,
                                    training_data=st.session_state.transformed_data.get("skill_matrix"),
                                    description=f"NMF model (preprocessing={use_preprocessing}, tuning={use_tuning})"
                                )

                                if model_id:
                                    st.success(f"✅ MLモデル学習が完了し、保存されました（ID: {model_id[:8]}...）")
                                else:
                                    st.success("✅ MLモデル学習が完了しました。")
                            except Exception as save_error:
                                st.warning(f"⚠️ モデルの保存に失敗しましたが、モデルは使用可能です: {save_error}")
                else:
                    st.success("✅ MLモデル学習が完了しました。")
                    st.info("💡 ログインするとモデルを保存して再利用できます。")

                st.rerun()
            except Exception as e:
                import traceback
                import sys
                from io import StringIO

                # stdoutとstderrを復元（エラー時に復元されていない場合に備えて）
                if hasattr(sys.stdout, 'getvalue'):
                    try:
                        captured_stdout = sys.stdout.getvalue()
                        captured_stderr = sys.stderr.getvalue() if hasattr(sys.stderr, 'getvalue') else ""

                        if captured_stdout:
                            debug_messages.append(f"\n--- 標準出力 (stdout) [エラー前] ---")
                            debug_messages.append(captured_stdout)
                        if captured_stderr:
                            debug_messages.append(f"\n--- エラー出力 (stderr) [エラー前] ---")
                            debug_messages.append(captured_stderr)
                    except:
                        pass

                # エラー情報をデバッグメッセージに追加
                debug_messages.append(f"\n❌ エラー発生")
                debug_messages.append(f"   - エラータイプ: {type(e).__name__}")
                debug_messages.append(f"   - エラーメッセージ: {e}")
                debug_messages.append(f"\n--- トレースバック ---")
                debug_messages.append(traceback.format_exc())
                st.session_state.debug_messages = debug_messages.copy()

                st.error(f"❌ エラーが発生しました: {type(e).__name__}: {e}")
                st.code(traceback.format_exc())
                st.info("デバッグ情報:")
                st.write("transformed_data keys:", list(st.session_state.transformed_data.keys()))

                # デバッグ情報を表示
                with st.expander("🔍 キャプチャされた出力", expanded=True):
                    st.code("\n".join(debug_messages))

                # エラー時もデバッグ情報を表示
                st.warning("⚠️ 詳細なデバッグ情報は上記のセクションに保存されています。ページをリロードしても情報は残ります。")


# =========================================================
# 学習結果の分析
# =========================================================
if st.session_state.get("model_trained", False):
    st.markdown("---")
    st.subheader("📊 学習結果の分析")

    recommender = st.session_state.ml_recommender
    mf_model = recommender.mf_model

    # 目的関数の明確な表示（ハイパーパラメータチューニング時）
    if hasattr(recommender, 'tuning_results') and recommender.tuning_results is not None:
        tuner = recommender.tuning_results['tuner']
        st.markdown("---")
        st.markdown("### 🎯 最適化の目的関数")
        st.success(
            f"""
            ## {tuner.objective_description}

            **説明：** NMFモデルは、複数の異なるランダム初期値から開始した学習を、K-fold交差検証により評価し、
            最も再構成誤差が低い（＝データをより正確に復元できる）パラメータセットを自動的に見つけます。

            **正規化再構成誤差とは：** 学習データに対する復元精度を、データスケール非依存な相対的な誤差として計算したもので、
            値が小さいほどモデルがメンバー×力量マトリクスをより正確に分解・復元できていることを示します。
            """
        )
        st.markdown("---")

        # Optunaチューニング結果の可視化
        st.markdown("### 📊 ハイパーパラメータチューニング結果の可視化")

        try:
            study = tuner.study

            # Pruning統計情報
            pruning_stats = get_pruned_trials_count(study)

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("総試行数", pruning_stats['total'])
            with col_stat2:
                st.metric("完了試行数", pruning_stats['complete'])
            with col_stat3:
                st.metric("枝刈り数", pruning_stats['pruned'])
            with col_stat4:
                st.metric("枝刈り率", f"{pruning_stats['pruning_rate']:.1f}%")

            if pruning_stats['pruned'] > 0:
                st.info(
                    f"💡 **Pruning（枝刈り）効果**: {pruning_stats['pruned']}個の有望でない試行を早期終了することで、"
                    f"計算時間を約{pruning_stats['pruning_rate']:.0f}%削減しました。"
                )

            # 上位試行のサマリーテーブル
            with st.expander("🏆 上位10試行の詳細", expanded=False):
                best_trials_df = get_best_trials_summary(study, top_n=10)
                st.dataframe(best_trials_df, use_container_width=True)
                st.download_button(
                    label="📥 上位試行データをダウンロード（CSV）",
                    data=best_trials_df.to_csv(index=False).encode('utf-8-sig'),
                    file_name="optuna_best_trials.csv",
                    mime="text/csv",
                )

            # Optunaの可視化グラフ生成
            st.markdown("#### 📈 チューニング過程の詳細分析")

            # パラメータリストを取得
            param_names = list(study.best_params.keys())

            with st.spinner("可視化グラフを生成中..."):
                visualizations = generate_optuna_visualizations(study, params_to_plot=param_names)

            # 6種類のグラフをタブで表示
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 最適化履歴",
                "🎯 パラメータ重要度",
                "🔗 パラレル座標",
                "🗺️ 等高線図",
                "📊 スライス",
                "📉 経験分布関数"
            ])

            with tab1:
                if 'optimization_history' in visualizations:
                    st.plotly_chart(visualizations['optimization_history'], use_container_width=True)
                    st.markdown("""
                    **最適化履歴**: 各試行の目的関数値の推移を表示します。
                    青線は各試行の値、赤線は最良値の更新を示します。
                    """)
                else:
                    st.warning("⚠️ 最適化履歴の生成に失敗しました")

            with tab2:
                if 'param_importances' in visualizations:
                    st.plotly_chart(visualizations['param_importances'], use_container_width=True)
                    st.markdown("""
                    **パラメータ重要度**: 各ハイパーパラメータが目的関数に与える影響度を表示します。
                    重要度が高いパラメータほど、最適化において重要な役割を果たしています。
                    """)
                else:
                    st.warning("⚠️ パラメータ重要度の生成に失敗しました")

            with tab3:
                if 'parallel_coordinate' in visualizations:
                    st.plotly_chart(visualizations['parallel_coordinate'], use_container_width=True)
                    st.markdown("""
                    **パラレル座標プロット**: すべてのパラメータと目的関数の関係を同時に可視化します。
                    各線は1つの試行を表し、色は目的関数値を示します（青=良い、赤=悪い）。
                    """)
                else:
                    st.warning("⚠️ パラレル座標プロットの生成に失敗しました")

            with tab4:
                if 'contour' in visualizations:
                    st.plotly_chart(visualizations['contour'], use_container_width=True)
                    st.markdown("""
                    **等高線図**: 2つのパラメータ間の相互作用と目的関数値の関係を表示します。
                    色が濃い領域ほど目的関数値が低い（良い）ことを示します。
                    """)
                else:
                    st.warning("⚠️ 等高線図の生成に失敗しました")

            with tab5:
                if 'slice' in visualizations:
                    st.plotly_chart(visualizations['slice'], use_container_width=True)
                    st.markdown("""
                    **スライスプロット**: 各パラメータが目的関数に与える個別の影響を表示します。
                    他のパラメータを固定した状態で、1つのパラメータのみを変化させた場合の効果を確認できます。
                    """)
                else:
                    st.warning("⚠️ スライスプロットの生成に失敗しました")

            with tab6:
                if 'edf' in visualizations:
                    st.plotly_chart(visualizations['edf'], use_container_width=True)
                    st.markdown("""
                    **経験分布関数（EDF）**: 目的関数値の累積分布を表示します。
                    探索がどの範囲の値に集中しているかを確認できます。
                    """)
                else:
                    st.warning("⚠️ 経験分布関数の生成に失敗しました")

        except Exception as viz_error:
            st.error(f"❌ 可視化の生成中にエラーが発生しました: {viz_error}")
            import traceback
            st.code(traceback.format_exc())

        st.markdown("---")

    # 基本統計
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("潜在因子数", mf_model.n_components)

    with col2:
        st.metric("メンバー数", len(mf_model.member_index))

    with col3:
        st.metric("力量数", len(mf_model.competence_index))

    with col4:
        error = mf_model.get_reconstruction_error()
        st.metric("再構成誤差", f"{error:.4f}")

    # NMF成分の分析
    st.markdown("### 🔍 NMF潜在因子の分析")

    st.markdown(
        "NMFはメンバー×力量マトリクスを**メンバー因子行列**と**力量因子行列**に分解します。\n"
        "各潜在因子は、特定の力量群（スキルセット）を表し、メンバーはこれらの因子の組み合わせで表現されます。"
    )

    # 各潜在因子の特徴を分析
    with st.expander("📈 潜在因子ごとの代表力量（トップ10）"):
        competence_master = st.session_state.transformed_data["competence_master"]

        # 表示対象の潜在因子数を選択
        col_factor_select1, col_factor_select2 = st.columns(2)
        with col_factor_select1:
            show_all_factors = st.checkbox(
                "すべての潜在因子を表示",
                value=False,
                help="チェックするとすべての潜在因子を表示します。未チェック時は最初の10個を表示"
            )

        n_factors_to_show = mf_model.n_components if show_all_factors else min(10, mf_model.n_components)

        for factor_idx in range(n_factors_to_show):
            st.markdown(f"#### 潜在因子 {factor_idx + 1}")

            # この因子で重みが高い力量を取得
            factor_weights = mf_model.H[factor_idx, :]

            # 非ゼロの重みを持つ力量のみを取得
            non_zero_indices = np.where(factor_weights > 1e-10)[0]

            if len(non_zero_indices) > 0:
                # 非ゼロの力量から上位10個を選択
                non_zero_weights = factor_weights[non_zero_indices]
                top_local_indices = non_zero_weights.argsort()[-10:][::-1]
                top_indices = non_zero_indices[top_local_indices]
            else:
                # すべてが0に近い場合（潜在因子が不要）
                st.warning(f"⚠️ 潜在因子 {factor_idx + 1} はすべて 0 または 0 に非常に近い値です。このモデルでは不使用の潜在因子と考えられます。")
                st.info("💡 潜在因子数（n_components）が多すぎる可能性があります。減らしてモデルを再学習することをお勧めします。")
                continue

            top_competences = [mf_model.competence_codes[i] for i in top_indices]
            top_weights = [factor_weights[i] for i in top_indices]

            # 力量名を取得
            top_competence_names = []
            for comp_code in top_competences:
                comp_info = competence_master[competence_master["力量コード"] == comp_code]
                if len(comp_info) > 0:
                    top_competence_names.append(comp_info.iloc[0]["力量名"])
                else:
                    top_competence_names.append(comp_code)

            # データフレームで表示（重みが大きい順に上から表示するため、降順でソート）
            df_factor = pd.DataFrame({
                "力量名": top_competence_names,
                "重み": top_weights
            }).sort_values("重み", ascending=False).reset_index(drop=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                # 棒グラフ（重みが大きい順に上から表示）
                fig = px.bar(
                    df_factor,
                    x="重み",
                    y="力量名",
                    orientation="h",
                    title=f"潜在因子 {factor_idx + 1} の代表力量（上ほど重みが大きい）"
                )
                # y軸の順序を逆にして、重みが大きいものが上に来るようにする
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # テーブル（重みが大きい順に表示）
                st.dataframe(df_factor, use_container_width=True, height=400)

    # メンバーの潜在因子分布
    with st.expander("👥 メンバーの潜在因子分布"):
        st.markdown("各メンバーがどの潜在因子を強く持っているかを示します。")

        # ランダムに10名をサンプル
        import numpy as np

        n_members_to_show = min(10, len(mf_model.member_codes))
        random_indices = np.random.choice(len(mf_model.member_codes), n_members_to_show, replace=False)

        member_codes = [mf_model.member_codes[i] for i in random_indices]
        member_names = []
        members_df = st.session_state.transformed_data["members_clean"]
        for code in member_codes:
            member_info = members_df[members_df["メンバーコード"] == code]
            if len(member_info) > 0:
                member_names.append(member_info.iloc[0]["メンバー名"])
            else:
                member_names.append(code)

        # 各メンバーの潜在因子の重みを取得（メンバー名とメンバーコードの両方を含める）
        member_factors_data = []
        for i, (idx, member_code) in enumerate(zip(random_indices, member_codes)):
            factors = mf_model.W[idx, :]
            for factor_idx, weight in enumerate(factors):
                member_factors_data.append({
                    "メンバー名": member_names[i],
                    "メンバーコード": member_code,
                    "潜在因子": f"因子{factor_idx + 1}",
                    "重み": weight
                })

        df_member_factors = pd.DataFrame(member_factors_data)

        # タブで2パターンを切り替え
        tab1, tab2 = st.tabs(["📝 メンバー名で表示", "🔢 メンバーコードで表示"])

        with tab1:
            # メンバー名でのヒートマップ
            # 重複チェック
            duplicates = df_member_factors[df_member_factors.duplicated(subset=["メンバー名", "潜在因子"], keep=False)]
            if not duplicates.empty:
                st.warning(f"⚠️ 重複データが検出されました（{len(duplicates)}件）。重複を削除します。")
                df_member_factors_name = df_member_factors.drop_duplicates(subset=["メンバー名", "潜在因子"], keep="first")
            else:
                df_member_factors_name = df_member_factors.copy()

            pivot_table_name = df_member_factors_name.pivot_table(
                index="メンバー名",
                columns="潜在因子",
                values="重み",
                aggfunc="mean"
            )

            fig_name = px.imshow(
                pivot_table_name,
                labels=dict(x="潜在因子", y="メンバー名", color="重み"),
                title="メンバーの潜在因子分布ヒートマップ（メンバー名）",
                color_continuous_scale="Blues"
            )
            fig_name.update_layout(height=500)
            st.plotly_chart(fig_name, use_container_width=True)

        with tab2:
            # メンバーコードでのヒートマップ
            # 重複チェック
            duplicates_code = df_member_factors[df_member_factors.duplicated(subset=["メンバーコード", "潜在因子"], keep=False)]
            if not duplicates_code.empty:
                st.warning(f"⚠️ 重複データが検出されました（{len(duplicates_code)}件）。重複を削除します。")
                df_member_factors_code = df_member_factors.drop_duplicates(subset=["メンバーコード", "潜在因子"], keep="first")
            else:
                df_member_factors_code = df_member_factors.copy()

            # メンバーコードを文字列型として明示的に変換
            df_member_factors_code["メンバーコード"] = df_member_factors_code["メンバーコード"].astype(str)

            pivot_table_code = df_member_factors_code.pivot_table(
                index="メンバーコード",
                columns="潜在因子",
                values="重み",
                aggfunc="mean"
            )

            # go.Heatmapを使用してカテゴリデータとして扱う
            import plotly.graph_objects as go

            fig_code = go.Figure(data=go.Heatmap(
                z=pivot_table_code.values,
                x=pivot_table_code.columns.tolist(),
                y=pivot_table_code.index.tolist(),
                colorscale="Blues",
                colorbar=dict(title="重み"),
                hoverongaps=False,
                hovertemplate="メンバーコード: %{y}<br>潜在因子: %{x}<br>重み: %{z:.3f}<extra></extra>"
            ))

            fig_code.update_layout(
                title="メンバーの潜在因子分布ヒートマップ（メンバーコード）",
                xaxis_title="潜在因子",
                yaxis_title="メンバーコード",
                height=500,
                yaxis=dict(type='category')  # カテゴリデータとして扱う
            )
            st.plotly_chart(fig_code, use_container_width=True)

    # 力量の潜在因子分布
    with st.expander("💡 力量の潜在因子分布"):
        st.markdown("各力量がどの潜在因子に関連しているかを示します。")

        # ランダムに10個の力量をサンプル
        n_competences_to_show = min(10, len(mf_model.competence_codes))
        random_comp_indices = np.random.choice(len(mf_model.competence_codes), n_competences_to_show, replace=False)

        competence_codes = [mf_model.competence_codes[i] for i in random_comp_indices]
        competence_names = []
        for code in competence_codes:
            comp_info = competence_master[competence_master["力量コード"] == code]
            if len(comp_info) > 0:
                competence_names.append(comp_info.iloc[0]["力量名"])
            else:
                competence_names.append(code)

        # 各力量の潜在因子の重みを取得
        competence_factors_data = []
        for i, (idx, comp_code) in enumerate(zip(random_comp_indices, competence_codes)):
            factors = mf_model.H[:, idx]
            for factor_idx, weight in enumerate(factors):
                competence_factors_data.append({
                    "力量": competence_names[i],  # インデックスで直接参照
                    "潜在因子": f"因子{factor_idx + 1}",
                    "重み": weight
                })

        df_competence_factors = pd.DataFrame(competence_factors_data)

        # 重複チェックとデバッグ情報
        duplicates_comp = df_competence_factors[df_competence_factors.duplicated(subset=["力量", "潜在因子"], keep=False)]
        if not duplicates_comp.empty:
            st.warning(f"⚠️ 重複データが検出されました（{len(duplicates_comp)}件）。重複を削除します。")
            df_competence_factors = df_competence_factors.drop_duplicates(subset=["力量", "潜在因子"], keep="first")

        # ヒートマップ
        pivot_table_comp = df_competence_factors.pivot_table(
            index="力量",
            columns="潜在因子",
            values="重み",
            aggfunc="mean"  # 万が一重複がある場合は平均を取る
        )

        fig = px.imshow(
            pivot_table_comp,
            labels=dict(x="潜在因子", y="力量", color="重み"),
            title="力量の潜在因子分布ヒートマップ",
            color_continuous_scale="Greens"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # モデル評価指標
    with st.expander("📉 モデル評価指標"):
        st.markdown("### 再構成誤差の詳細")

        error = mf_model.get_reconstruction_error()
        normalized_error = mf_model.get_normalized_reconstruction_error()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("再構成誤差（Frobenius ノルム）", f"{error:.6f}")
        with col2:
            st.metric("正規化再構成誤差（相対誤差）", f"{normalized_error:.6f}")

        st.info("""
        **メトリクス解説:**
        - **再構成誤差（Frobenius ノルム）**: モデルが元のデータを再構成する際の絶対的な誤差
        - **正規化再構成誤差**: データのスケールに依存しない相対的な誤差（異なるデータセット間での比較に有用）
        """)

        # 評価基準と改善提案
        if normalized_error < 0.1:
            st.success("✅ **非常に良好なモデルです**")
            st.markdown("正規化再構成誤差が0.1以下で、モデルは元のデータを非常によく再現しています。")
        elif normalized_error < 0.2:
            st.success("✅ **良好なモデルです**")
            st.markdown("正規化再構成誤差が0.2以下で、モデルは元のデータをよく再現しています。")
        elif normalized_error < 0.3:
            st.warning("⚠️ **許容範囲ですが、改善の余地があります**")
            st.markdown("正規化再構成誤差が0.3以下で許容範囲内ですが、さらなる改善が可能です。")
        else:
            st.error("❌ **改善が必要です**")
            st.markdown("正規化再構成誤差が0.3以上で、モデルの精度向上が推奨されます。")

        # モデルスパース性の分析
        st.markdown("---")
        st.markdown("### 📊 モデルスパース性（疎行列性）")

        sparsity_info = mf_model.get_model_sparsity()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("メンバー因子（W）のスパース性", f"{sparsity_info['W_sparsity']:.2f}%")
        with col2:
            st.metric("力量因子（H）のスパース性", f"{sparsity_info['H_sparsity']:.2f}%")

        # 診断メッセージ
        st.markdown(sparsity_info['recommendation'])

        if sparsity_info['unused_factors']:
            st.warning(f"""
            **⚠️ 未使用の潜在因子が検出されました**

            未使用の潜在因子インデックス: {sparsity_info['unused_factors']}

            現在の潜在因子数を {mf_model.n_components - len(sparsity_info['unused_factors'])} に削減して再学習することをお勧めします。
            これにより計算効率が向上し、モデルの解釈性が向上します。
            """)

        # 追加メトリクス
        st.markdown("---")
        st.markdown("### 📈 学習情報")

        col1, col2, col3 = st.columns(3)

        with col1:
            n_iter = mf_model.actual_n_iter_ if mf_model.actual_n_iter_ is not None else getattr(mf_model.model, 'n_iter_', 'N/A')
            st.metric("イテレーション数", n_iter)

        with col2:
            st.metric("潜在因子数", mf_model.n_components)

        with col3:
            st.metric("学習データ点数", len(mf_model.member_codes) * len(mf_model.competence_codes))

        # 改善提案
        if normalized_error >= 0.2:
            st.markdown("---")
            st.markdown("### 💡 改善提案")

            current_components = mf_model.n_components

            st.info(f"""
            **推奨される改善策:**

            1. **ハイパーパラメータチューニング**:
               - 上記の「学習オプション」で「ハイパーパラメータチューニング (Optuna)」を有効にしてモデルを再学習してください
               - ベイズ最適化により最適なパラメータが自動的に探索されます

            2. **データ前処理の有効化**:
               - 「データ前処理を使用」を有効にすることで、外れ値の除去と正規化が行われます
               - スパースなデータに対して特に効果的です

            3. **手動でのパラメータ調整** (config.py):
               - 潜在因子数: 現在 {current_components} → 25〜35 に増加を検討
               - 正則化強度: alpha_W, alpha_H を 0.05〜0.1 に調整
               - 最大イテレーション数: max_iter を 1500〜2000 に増加

            詳細は `docs/EVALUATION.md` を参照してください。
            """)

    # ハイパーパラメータチューニング結果の表示
    if recommender.tuning_results is not None:
        with st.expander("🎯 ハイパーパラメータチューニング結果", expanded=True):
            tuning_results = recommender.tuning_results
            tuner = tuning_results['tuner']

            # 目的関数の説明
            st.markdown("### 🎯 最適化の目的")
            st.info(
                f"""
                **{tuner.objective_description}**

                **探索対象パラメータ:**
                - **n_components**: 潜在因子数（10-30の範囲）
                - **alpha_W**: メンバー因子の正則化強度（0.001-0.5、対数スケール）
                - **alpha_H**: 力量因子の正則化強度（0.001-0.5、対数スケール）
                - **l1_ratio**: L1正則化の割合（0.0-1.0）
                - **n_init**: 初期化試行回数（1-5、複数初期値から最良を選択）

                **自動設定パラメータ:**
                - **max_iter**: データ特性から自動計算（{tuner.max_iter}）
                  - データサイズ、スパース性、Early Stopping設定に基づいて決定

                **固定パラメータ:**
                - init=nndsvda, solver=cd, tol=1e-5

                **改善機能:**
                - Early Stoppingが検証セットの誤差を監視（過学習防止）
                - 訓練誤差と検証誤差の乖離を記録（汎化ギャップ監視）
                """
            )

            st.markdown("---")
            st.markdown("### 📊 チューニングサマリー")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### デフォルトパラメータ")
                default_params = tuning_results['default_params']
                for key, value in default_params.items():
                    if key in ['n_components', 'max_iter', 'alpha_W', 'alpha_H', 'l1_ratio', 'n_init']:
                        st.text(f"{key}: {value}")

            with col2:
                st.markdown("#### 最適パラメータ")
                best_params = tuning_results['best_params']
                for key, value in best_params.items():
                    if isinstance(value, float):
                        st.text(f"{key}: {value:.4f}")
                    else:
                        st.text(f"{key}: {value}")

                # 最良トライアルのrandom_stateを表示
                tuner = tuning_results['tuner']
                best_trial = tuner.study.best_trial
                best_random_state = best_trial.user_attrs.get('random_state', 'N/A')
                st.text(f"random_state: {best_random_state}")

            st.markdown("---")
            st.markdown("### 📈 最適化履歴")

            # 最適化履歴をプロット
            tuner = tuning_results['tuner']

            try:
                fig_history = tuner.plot_optimization_history()
                if fig_history:
                    st.plotly_chart(fig_history, use_container_width=True)

                st.markdown("### 🔍 パラメータの重要度")
                fig_importance = tuner.plot_param_importances()
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)

                st.info("""
                **パラメータの重要度**は、各パラメータが再構成誤差に与える影響の大きさを示しています。
                重要度が高いパラメータほど、モデル性能に大きく影響します。
                """)

            except Exception as e:
                st.warning(f"グラフの表示中にエラーが発生しました: {e}")

            # パラメータの探索範囲統計
            st.markdown("### 📊 パラメータ探索範囲の統計")
            try:
                trials_df = tuner.get_optimization_history()

                # 各パラメータの統計情報を計算
                # max_iterは自動計算されるため除外
                param_stats = {}
                param_cols = ['params_n_components', 'params_alpha_W', 'params_alpha_H',
                             'params_l1_ratio']

                stats_data = []
                for col in param_cols:
                    if col in trials_df.columns:
                        param_name = col.replace('params_', '')
                        stats_data.append({
                            'パラメータ': param_name,
                            '最小値': f"{trials_df[col].min():.6f}",
                            '最大値': f"{trials_df[col].max():.6f}",
                            '平均値': f"{trials_df[col].mean():.6f}",
                            '標準偏差': f"{trials_df[col].std():.6f}"
                        })

                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)

                    st.info("""
                    **探索範囲の統計**は、Optunaが実際に試したパラメータの範囲を示しています。
                    - **最小値・最大値**：実際に試された値の範囲
                    - **標準偏差**が大きい：広い範囲を探索している（良い兆候）
                    - **標準偏差**が小さい：狭い範囲に集中している（探索が不十分な可能性）
                    """)

                # パラメータ分布のヒストグラムを表示
                with st.expander("📊 パラメータ分布のヒストグラム"):
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    # alpha_W と alpha_H のヒストグラムを作成（対数スケール重要）
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('alpha_W (対数スケール)', 'alpha_H (対数スケール)',
                                       'l1_ratio', 'n_components')
                    )

                    if 'params_alpha_W' in trials_df.columns:
                        fig.add_trace(
                            go.Histogram(x=trials_df['params_alpha_W'], name='alpha_W', nbinsx=20),
                            row=1, col=1
                        )
                        fig.update_xaxes(type="log", row=1, col=1)

                    if 'params_alpha_H' in trials_df.columns:
                        fig.add_trace(
                            go.Histogram(x=trials_df['params_alpha_H'], name='alpha_H', nbinsx=20),
                            row=1, col=2
                        )
                        fig.update_xaxes(type="log", row=1, col=2)

                    if 'params_l1_ratio' in trials_df.columns:
                        fig.add_trace(
                            go.Histogram(x=trials_df['params_l1_ratio'], name='l1_ratio', nbinsx=20),
                            row=2, col=1
                        )

                    if 'params_n_components' in trials_df.columns:
                        fig.add_trace(
                            go.Histogram(x=trials_df['params_n_components'], name='n_components', nbinsx=20),
                            row=2, col=2
                        )

                    fig.update_layout(height=600, showlegend=False, title_text="パラメータ分布（全トライアル）")
                    st.plotly_chart(fig, use_container_width=True)

                    st.info("""
                    **ヒストグラム**で、Optunaが各パラメータをどれだけ広く探索したか確認できます。
                    - 対数スケールのパラメータ（alpha_W, alpha_H）は広い範囲（0.001～1.0）を探索
                    - 分布が偏っている場合、探索範囲の調整が必要な可能性
                    """)

            except Exception as e:
                st.warning(f"統計情報の計算中にエラーが発生しました: {e}")

            # 詳細な試行結果を表示
            with st.expander("📋 全試行の詳細結果"):
                try:
                    trials_df = tuner.get_optimization_history()
                    # 必要な列のみを表示（random_stateも追加）
                    display_cols = ['number', 'value', 'params_n_components', 'params_alpha_W',
                                   'params_alpha_H', 'params_l1_ratio', 'params_max_iter',
                                   'user_attrs_random_state', 'user_attrs_n_iter', 'state']
                    available_cols = [col for col in display_cols if col in trials_df.columns]

                    if available_cols:
                        # 再構成誤差でソートして表示
                        display_df = trials_df[available_cols].sort_values('value')
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=400
                        )

                        # 最良トライアルの詳細を強調表示
                        best_trial_num = display_df.iloc[0]['number']
                        best_value = display_df.iloc[0]['value']
                        st.success(f"✨ 最良トライアル: #{int(best_trial_num)} (再構成誤差: {best_value:.6f})")

                    else:
                        st.dataframe(trials_df, use_container_width=True, height=400)

                    st.info("""
                    **user_attrs_random_state**: 各トライアルで使用されたrandom_state（異なる値で探索）
                    **user_attrs_n_iter**: 収束までのイテレーション数
                    """)

                except Exception as e:
                    st.warning(f"試行結果の表示中にエラーが発生しました: {e}")

    # 訓練 vs テスト誤差の評価
    if recommender.tuning_results is not None and hasattr(recommender.tuning_results.get('tuner'), 'evaluate_training_vs_test'):
        tuner = recommender.tuning_results.get('tuner')
        if hasattr(tuner, 'test_matrix') and tuner.test_matrix is not None:
            with st.expander("📊 訓練 vs テスト誤差の分析（汎化性能診断）"):
                st.markdown("### モデルの汎化性能を診断します")

                try:
                    # 訓練 vs テスト誤差を計算
                    eval_results = tuner.evaluate_training_vs_test(mf_model)

                    # 結果を表示
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("訓練データ誤差", f"{eval_results['train_error']:.6f}")
                        st.text(f"サイズ: {eval_results['train_size']}")

                    with col2:
                        if 'test_error' in eval_results:
                            st.metric("テストデータ誤差", f"{eval_results['test_error']:.6f}")
                            st.text(f"サイズ: {eval_results['test_size']}")
                        else:
                            st.metric("テストデータ誤差", "N/A")

                    with col3:
                        if 'generalization_gap' in eval_results:
                            gap = eval_results['generalization_gap']
                            st.metric("汎化ギャップ", f"{gap:.6f}")
                            st.text(f"差分比: {(gap/eval_results['train_error']*100):.1f}%")

                    st.markdown("---")
                    st.markdown("### 診断結果")

                    # 診断メッセージを表示
                    diagnosis = eval_results.get('diagnosis', '')
                    if '優れた' in diagnosis or '✅' in diagnosis:
                        st.success(diagnosis)
                    elif '軽度' in diagnosis or '⚠️' in diagnosis:
                        st.warning(diagnosis)
                    else:
                        st.error(diagnosis)

                    # 詳細説明
                    st.info("""
                    **汎化ギャップの解釈:**
                    - **ギャップが小さい（<10%）**: モデルの汎化性能が優れている
                    - **ギャップが中程度（10-30%）**: 軽度の過学習が見られるが許容範囲
                    - **ギャップが大きい（>30%）**: 顕著な過学習の可能性、モデルの改善推奨

                    **改善方法:**
                    1. 正則化強度（alpha_W, alpha_H）を増加させる
                    2. 早期停止（Early Stopping）を有効にする
                    3. データ前処理を有効にする
                    4. より多くの訓練データを用意する
                    """)

                except Exception as e:
                    st.warning(f"汎化性能の診断中にエラーが発生しました: {e}")
                    st.info("テストセットが分離されていないか、診断が利用できません。")

    st.markdown("---")
    st.success("✅ 学習結果の分析が完了しました。")
    st.info("👉 サイドバーから「推論」ページに移動して、推薦を実行してください。")
