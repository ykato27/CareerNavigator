import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
import graphviz

from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
from skillnote_recommendation.graph.causal_graph_visualizer import CausalGraphVisualizer
from skillnote_recommendation.utils.ui_components import (
    apply_enterprise_styles,
    render_page_header
)

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - 因果推論推薦",
    page_icon="🧭",
    layout="wide"
)

apply_enterprise_styles()

render_page_header(
    title="因果推論推薦",
    icon="🔗",
    description="データからスキル間の因果関係を発見し、説得力のある推薦を行います"
)

# =========================================================
# データチェック
# =========================================================
if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
    st.warning("まずはトップページでデータを読み込んでください。")
    st.stop()

td = st.session_state.transformed_data

# =========================================================
# 因果推論推薦の仕組み説明
# =========================================================
st.markdown("---")
st.subheader("🔍 因果推論推薦の仕組み")

with st.expander("💡 この機能で実際に行っていること", expanded=True):
    st.markdown("""
    このページでは、**因果推論**と**ベイジアンネットワーク**を組み合わせて、
    **説明可能で精度の高いスキル推薦**を実現しています。
    """)
    
    # 説明画像を表示
    st.image("assets/causal_logic_whiteboard.png", use_container_width=True)
    
    st.markdown("""
    ### 📊 3つの主要技術
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1️⃣ LiNGAM
        **因果構造の発見**
        
        - スキル間の「原因→結果」関係を発見
        - 相関ではなく**因果関係**を特定
        - 「Aを学ぶとBが習得しやすくなる」という関係を数値化
        
        **技術**: Linear Non-Gaussian Acyclic Model
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Bayesian Network
        **確率的推論**
        
        - 同じスキルパターンを持つ人の習得確率を計算
        - 条件付き確率で「あなたならこのスキルを習得する可能性」を評価
        
        **技術**: 階層ベイズモデル + MCMC
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Causal Score
        **3軸スコアリング**
        
        - **Readiness**: 今学べる準備ができているか
        - **Bayesian**: 統計的に習得可能性が高いか
        - **Utility**: 将来のキャリアに役立つか
        
        **統合**: 重み付き総合スコア
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔄 推薦の流れ
    
    1. **データ収集** → メンバーのスキル保有データ（0/1）を入力
    2. **因果構造学習（LiNGAM）** → スキル間の因果関係を自動発見
    3. **ベイジアンネットワーク構築** → 条件付き確率を学習
    4. **スコア計算** → Readiness、Bayesian、Utilityの3軸で評価
    5. **重み最適化** → ベイズ最適化で最適な重みを自動調整
    6. **推薦出力** → 総合スコア順にスキルを推薦
    
    ### 🎯 従来手法との違い
    
    | 項目 | 従来の推薦 | 本システム（因果推論推薦） |
    |---|---|---|
    | **手法** | 協調フィルタリング | 因果推論 + ベイジアンネットワーク |
    | **根拠** | 「似た人が学んでいる」 | 「Aを学ぶとBが習得しやすくなる（因果関係）」 |
    | **説明性** | ❌ ブラックボックス | ✅ 因果グラフで可視化 |
    | **個別最適化** | ⚠️ 弱い | ✅ Readinessで現在のスキルを考慮 |
    | **将来性考慮** | ❌ なし | ✅ Utilityで将来のキャリアパスを考慮 |
    
    ### 🧮 スコアの計算式
    
    ```
    総合スコア = Readiness × w₁ + Bayesian × w₂ + Utility × w₃
    
    ここで:
    - Readiness = Σ(保有スキル → 対象スキルの因果効果)
    - Bayesian = P(対象スキル=1 | 保有スキル) 【ベイジアンネットワークで計算】
    - Utility = Σ(対象スキル → 未習得スキルの因果効果)
    - w₁, w₂, w₃ = 重み（合計1.0、ベイズ最適化で自動調整可能）
    ```
    
    デフォルト重み: **Readiness 60%、Bayesian 30%、Utility 10%**
    """)

st.markdown("---")
st.subheader("🧠 因果モデルの学習")

with st.expander("設定と学習", expanded=not st.session_state.get("causal_model_trained", False)):
    st.markdown("""
    **LiNGAM (Linear Non-Gaussian Acyclic Model)** を用いて、スキル間の因果構造を学習します。

    - **クラスタリング**: 計算コスト削減のため、スキルを相関の高いグループに分割して処理します。
    - **因果探索**: 各グループ内で因果の向き（原因→結果）を特定します。
    """)

    col1, col2 = st.columns(2)
    with col1:
        min_members = st.number_input(
            "最小メンバー数/スキル",
            min_value=3,
            value=5,
            help="これより少ないメンバーしか持っていないスキルは除外します"
        )

    with col2:
        corr_threshold = st.slider(
            "クラスタリング相関閾値",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="この値以上の相関があるスキル同士を同じグループにします"
        )

    # 重み設定方法の選択
    st.markdown("---")
    st.markdown("### ⚙️ 推薦スコアの重み設定")

    weight_mode = st.radio(
        "重みの設定方法を選択",
        ["デフォルト重み（推奨）", "手動で重みを指定", "学習後に自動最適化"],
        help="デフォルトは Readiness:60%, Bayesian:30%, Utility:10%"
    )

    initial_weights = {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}
    run_optimization_after = False

    if weight_mode == "手動で重みを指定":
        st.markdown("**スライダーで初期重みを設定**")
        col_w1, col_w2, col_w3 = st.columns(3)

        with col_w1:
            readiness_w = st.slider(
                "Readiness（準備度）",
                0.0, 1.0, 0.6, 0.05,
                key="init_readiness"
            )
        with col_w2:
            bayesian_w = st.slider(
                "Bayesian（確率）",
                0.0, 1.0, 0.3, 0.05,
                key="init_bayesian"
            )
        with col_w3:
            utility_w = st.slider(
                "Utility（将来性）",
                0.0, 1.0, 0.1, 0.05,
                key="init_utility"
            )

        total_w = readiness_w + bayesian_w + utility_w
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"⚠️ 合計: {total_w:.2f}（適用時に正規化されます）")

        initial_weights = {
            'readiness': readiness_w,
            'bayesian': bayesian_w,
            'utility': utility_w
        }

    elif weight_mode == "学習後に自動最適化":
        st.info("💡 モデル学習後、ベイズ最適化で自動的に最適な重みを探索します（数分かかります）")
        run_optimization_after = True

        # 最適化パラメータ
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            opt_trials = st.number_input(
                "最適化試行回数",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key="init_opt_trials"
            )
        with col_opt2:
            opt_jobs_option = st.selectbox(
                "並列ジョブ数",
                options=["全コア使用（推奨）", "1", "2", "4", "8", "16"],
                index=0,
                key="init_opt_jobs",
                help="並列実行するジョブの数"
            )
            # 選択肢を数値に変換
            if opt_jobs_option == "全コア使用（推奨）":
                opt_jobs = -1
            else:
                opt_jobs = int(opt_jobs_option)

    if st.button("🚀 因果モデルを学習開始", type="primary"):
        with st.spinner("因果構造を学習中... (これには数分かかる場合があります)"):
            try:
                recommender = CausalGraphRecommender(
                    member_competence=td["member_competence"],
                    competence_master=td["competence_master"],
                    learner_params={
                        "correlation_threshold": corr_threshold,
                        "min_cluster_size": 3
                    },
                    weights=initial_weights
                )

                recommender.fit(min_members_per_skill=min_members)

                st.session_state.causal_recommender = recommender
                st.session_state.causal_model_trained = True
                st.success("✅ 因果構造の学習が完了しました！")

                # 自動最適化を実行
                if run_optimization_after:
                    st.info("🔄 重みの自動最適化を開始します...")
                    with st.spinner(f"ベイズ最適化を実行中... ({opt_trials}回の試行、並列処理で高速化)"):
                        try:
                            best_weights = recommender.optimize_weights(
                                n_trials=opt_trials,
                                n_jobs=opt_jobs,
                                holdout_ratio=0.2,
                                top_k=10
                            )
                            st.success(f"✅ 最適化完了！最適な重み: Readiness {best_weights['readiness']:.1%}, Bayesian {best_weights['bayesian']:.1%}, Utility {best_weights['utility']:.1%}")
                        except Exception as opt_error:
                            st.warning(f"⚠️ 最適化に失敗しました: {opt_error}")
                            st.info("デフォルト重みで続行します。")

                st.balloons()
                st.rerun()

            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")
                st.exception(e)

if not st.session_state.get("causal_model_trained", False):
    st.stop()

recommender = st.session_state.causal_recommender

# 後方互換性: 古いモデルにweights属性を追加
if not hasattr(recommender, 'weights'):
    recommender.weights = {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}
    st.warning("⚠️ モデルが古いバージョンです。デフォルトの重みを設定しました。最新機能を使うには、モデルを再学習してください。")

# 学習データのサマリー情報を表示
st.info(f"📊 学習済みモデル: メンバー数 {len(recommender.skill_matrix_.index)}人、スキル数 {len(recommender.skill_matrix_.columns)}個")

# =========================================================
# 重み最適化セクション
# =========================================================
st.markdown("---")
st.subheader("⚙️ 推薦スコアの重み調整")

with st.expander("💡 重みの最適化について", expanded=False):
    st.markdown("""
    推薦スコアは以下の3つの要素から計算されます：

    - **Readiness（準備度）**: 保有スキルから推奨スキルへの因果効果
    - **Bayesian（確率）**: 同様のスキルパターンを持つ人の習得確率
    - **Utility（将来性）**: 推奨スキルから将来のスキルへの因果効果

    これらの重みは、ベイズ最適化により自動調整できます。
    評価指標にはNDCG@K（推薦順位の精度）を使用します。
    """)

# 現在の重みを表示
current_weights = recommender.get_weights() if hasattr(recommender, 'get_weights') else recommender.weights

# 手動調整タブと自動最適化タブ
tab_adjust, tab_auto = st.tabs(["🎚️ 手動調整", "🤖 自動最適化"])

with tab_adjust:
    st.markdown("**スライダーで重みを調整し、推薦結果への影響をリアルタイムで確認できます**")

    col1, col2, col3 = st.columns(3)

    with col1:
        readiness_weight = st.slider(
            "Readiness（準備度）",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['readiness'],
            step=0.05,
            help="保有スキルから推奨スキルへの因果効果の重み"
        )

    with col2:
        bayesian_weight = st.slider(
            "Bayesian（確率）",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['bayesian'],
            step=0.05,
            help="同様のスキルパターンを持つ人の習得確率の重み"
        )

    with col3:
        utility_weight = st.slider(
            "Utility（将来性）",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['utility'],
            step=0.05,
            help="推奨スキルから将来のスキルへの因果効果の重み"
        )

    # 合計を表示
    total_weight = readiness_weight + bayesian_weight + utility_weight

    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ 重みの合計が {total_weight:.2f} です。適用時に自動的に正規化されます。")
    else:
        st.success(f"✅ 重みの合計: {total_weight:.2f}")

    # 適用ボタン
    if st.button("📝 この重みを適用", type="primary"):
        new_weights = {
            'readiness': readiness_weight,
            'bayesian': bayesian_weight,
            'utility': utility_weight
        }

        # 後方互換性: set_weightsメソッドがあれば使用、なければ直接設定
        if hasattr(recommender, 'set_weights'):
            recommender.set_weights(new_weights)
        else:
            total = sum(new_weights.values())
            recommender.weights = {k: v / total for k, v in new_weights.items()}

        st.success("✅ 重みを更新しました！下の推薦結果に反映されています。")
        st.rerun()

    # 現在の設定を表示
    st.info(f"**現在の重み**: Readiness {current_weights['readiness']:.1%} | Bayesian {current_weights['bayesian']:.1%} | Utility {current_weights['utility']:.1%}")

with tab_auto:
    st.markdown("**ベイズ最適化により、データから最適な重みを自動で探索します**")

    # 最適化設定
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        n_trials = st.number_input(
            "最適化試行回数",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="多いほど精度が上がりますが、時間がかかります"
        )
    with col_opt2:
        n_jobs_option = st.selectbox(
            "並列ジョブ数",
            options=["全コア使用（推奨）", "1", "2", "4", "8", "16"],
            index=0,
            help="並列実行するジョブの数"
        )
        # 選択肢を数値に変換
        if n_jobs_option == "全コア使用（推奨）":
            n_jobs = -1
        else:
            n_jobs = int(n_jobs_option)

    # 最適化実行ボタン
    if st.button("🎯 最適な重みを自動計算", type="primary"):
        # 後方互換性チェック
        if not hasattr(recommender, 'optimize_weights'):
            st.error("❌ 自動最適化機能は、新しいバージョンのモデルでのみ利用可能です。")
            st.warning("💡 因果モデルを再学習してください。")
        else:
            with st.spinner(f"ベイズ最適化を実行中... ({n_trials}回の試行、並列処理で高速化)"):
                try:
                    best_weights = recommender.optimize_weights(
                        n_trials=n_trials,
                        n_jobs=n_jobs,
                        holdout_ratio=0.2,
                        top_k=10
                    )

                    st.success("✅ 最適化が完了しました！")
                    st.balloons()

                    # 結果を表示
                    st.markdown("### 🎉 最適な重み")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric(
                            "Readiness",
                            f"{best_weights['readiness']:.1%}",
                            delta=f"{(best_weights['readiness'] - current_weights['readiness']):.1%}"
                        )
                    with col_r2:
                        st.metric(
                            "Bayesian",
                            f"{best_weights['bayesian']:.1%}",
                            delta=f"{(best_weights['bayesian'] - current_weights['bayesian']):.1%}"
                        )
                    with col_r3:
                        st.metric(
                            "Utility",
                            f"{best_weights['utility']:.1%}",
                            delta=f"{(best_weights['utility'] - current_weights['utility']):.1%}"
                        )

                    st.info("新しい重みが自動的に適用されました。下の推薦結果に反映されています。")

                except Exception as e:
                    st.error(f"最適化中にエラーが発生しました: {e}")
                    st.exception(e)

# =========================================================
# 推薦 & 可視化セクション
# =========================================================
st.markdown("---")

tab1, tab2 = st.tabs(["👤 メンバー別推薦", "🕸️ 因果グラフ全体"])

with tab1:
    st.subheader("メンバーへのスキル推薦")

    members = td["members_clean"]

    # 推薦可能なメンバーのみを選択肢として表示
    # (skill_matrix_に存在するメンバーコードのみ)
    available_members = recommender.skill_matrix_.index.tolist()
    member_options = [m for m in members["メンバーコード"].tolist() if m in available_members]

    if not member_options:
        st.warning("推薦可能なメンバーが見つかりません。学習データを確認してください。")
        st.stop()

    # メンバー選択
    selected_member_code = st.selectbox(
        "メンバーを選択",
        member_options,
        format_func=lambda x: f"{x} : {members[members['メンバーコード']==x]['氏名'].iloc[0] if '氏名' in members.columns else ''}"
    )

    if selected_member_code:
        st.markdown("### 🎯 推奨スキル（優先順位順）")
        
        # スコアの説明
        with st.expander("📖 スコアの見方", expanded=False):
            # 現在の重みを取得
            weights = recommender.get_weights() if hasattr(recommender, 'get_weights') else recommender.weights

            st.markdown(f"""
            推奨スコアは以下の3つの要素から計算されます:

            - **Readiness（準備度）**: 現在の保有スキルが、推奨スキルの習得をどれだけサポートするか
              - 高いほど、今すぐ学習を始めやすいスキル
              - 保有スキルから推奨スキルへの因果関係の強さで評価

            - **Bayesian（確率）**: 同様のスキルセットを持つ人が、そのスキルを習得している確率
              - 高いほど、あなたのようなスキルパターンの人が習得している可能性が高い
              - ベイジアンネットワークによる確率推論で評価

            - **Utility（将来性）**: 推奨スキルを習得することで、将来的にどれだけ多くのスキル習得が可能になるか
              - 高いほど、キャリアの選択肢を広げるスキル
              - 推奨スキルから他のスキルへの因果関係の強さで評価

            ---

            **現在の重み設定:**

            **総合スコア** = Readiness × {weights['readiness']:.1%} + Bayesian × {weights['bayesian']:.1%} + Utility × {weights['utility']:.1%}

            ※重みは「推薦スコアの重み調整」セクションで変更できます
            """)
        
        recommendations = recommender.recommend(selected_member_code, top_n=10)

        if not recommendations:
            # メンバーの保有スキル数を表示
            member_skills = recommender.skill_matrix_.loc[selected_member_code]
            owned_count = (member_skills > 0).sum()
            st.warning(f"💡 推奨できるスキルが見つかりませんでした。")
            st.info(f"現在の保有スキル数: {owned_count}個\n\n以下の可能性があります：\n- 既にほとんどのスキルを習得済み\n- 保有スキルと他のスキルの間に明確な因果関係が見つからなかった")
        else:
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    st.markdown(f"#### {i}. {rec['competence_name']}")

                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.metric("総合スコア", f"{rec['score']:.2f}")
                    with col2:
                        details = rec['details']
                        st.metric("準備度", f"{details['readiness_score_normalized']:.2f}")
                    with col3:
                        st.metric("確率", f"{details['bayesian_score_normalized']:.2f}")
                    with col4:
                        st.metric("将来性", f"{details['utility_score_normalized']:.2f}")
                    
                    
                    st.info(rec['explanation'])
                    
                    # 詳細な理由を表示
                    with st.expander("📋 詳細な推薦理由"):
                        details = rec['details']

                        st.markdown("**🟢 準備度（Readiness）**: なぜこのスキルが推奨されるか")
                        if details['readiness_reasons']:
                            st.markdown("あなたの以下の保有スキルが、このスキルの習得を後押しします:")
                            for skill, effect in details['readiness_reasons'][:5]:
                                st.write(f"- **{skill}** → 因果効果: {effect:.3f}")
                        else:
                            st.write("保有スキルからの直接的な因果関係は検出されませんでした。")

                        st.markdown("**🟣 確率（Bayesian）**: 同様のスキルパターンを持つ人の習得状況")
                        if details['bayesian_score'] > 0:
                            prob_pct = details['bayesian_score'] * 100
                            st.write(f"- あなたと同様のスキルセットを持つ方の **{prob_pct:.1f}%** がこのスキルを習得しています")
                        else:
                            st.write("ベイジアンネットワークによる確率推論ができませんでした。")

                        st.markdown("**🔵 将来性（Utility）**: このスキルを習得すると何ができるか")
                        if details['utility_reasons']:
                            st.markdown("このスキルを習得すると、以下のスキル習得がスムーズになります:")
                            for skill, effect in details['utility_reasons'][:5]:
                                st.write(f"- **{skill}** ← 因果効果: {effect:.3f}")
                        else:
                            st.write("将来のスキルへの直接的な因果関係は検出されませんでした。")
                    
                    st.markdown("---")
        
        # グラフ表示用の推奨スキル選択
        st.markdown("### 🔗 関連因果グラフ")
        st.caption("選択した推奨スキルを中心とした因果関係")
        
        # 推奨スキルから選択（上位10個まで）
        skill_options = [f"{i+1}. {rec['competence_name']} (スコア: {rec['score']:.2f})" 
                        for i, rec in enumerate(recommendations[:10])]
        selected_skill_idx = st.selectbox(
            "グラフを表示する推奨スキルを選択",
            range(min(10, len(recommendations))),
            format_func=lambda x: skill_options[x],
            help="上位10個の推奨スキルから選択できます。"
        )

        # 表示設定
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            graph_threshold = st.slider(
                "表示閾値",
                0.01, 1.0, 0.05, 0.01,
                key="ego_threshold",
                help="この値以上の因果係数を持つエッジのみ表示"
            )
        with col_g2:
            physics_enabled = st.checkbox(
                "物理演算",
                value=True,
                key="ego_physics",
                help="ノードの自動配置（重い場合はOFF推奨）"
            )
        with col_g3:
            show_negative_ego = st.checkbox(
                "負の因果も表示",
                value=False,
                key="ego_show_negative",
                help="赤線（負の因果関係）も表示する"
            )

        # エゴネットワークの可視化
        if recommendations:
            center_node = recommendations[selected_skill_idx]['competence_name']

            # Visualizer作成
            adj_matrix = recommender.learner.get_adjacency_matrix()
            visualizer = CausalGraphVisualizer(adj_matrix)

            # 保有スキルをハイライト用リストに
            member_skills_codes = td["member_competence"][
                td["member_competence"]["メンバーコード"] == selected_member_code
            ]["力量コード"].tolist()

            # コード -> 名前変換
            code_to_name = recommender.code_to_name
            member_skill_names = [code_to_name.get(c, c) for c in member_skills_codes]

            try:
                # エゴネットワークをインタラクティブに表示
                html_path = visualizer.visualize_ego_network_pyvis(
                    center_node=center_node,
                    radius=1,
                    threshold=graph_threshold,
                    show_negative=show_negative_ego,
                    member_skills=member_skill_names,
                    output_path="ego_network.html",
                    height="600px"
                )
                
                # HTMLファイルを読み込んで表示
                with open(html_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                components.html(source_code, height=600, scrolling=False)
                
                # 凡例を表示
                st.caption(f"💡 **{center_node}** を中心とした因果関係（拡大・移動可能）")
                st.caption(
                    "🟦 **青**: 推奨スキル（中心） | "
                    "🟩 **緑**: あなたの保有スキル（なぜ推奨されるか） | "
                    "⬜ **白**: 将来取得可能なスキル"
                )
            except Exception as e:
                st.error(f"グラフを描画できませんでした: {e}")

with tab2:
    st.subheader("因果グラフ全体像（インタラクティブ）")
    st.caption("学習されたスキル間の因果関係の全体像")

    # 表示設定パネル
    st.info(
        "📊 **因果関係の表示について**\n\n"
        "- **黒線（正の因果）**: スキルAを習得すると、スキルBの習得が促進される関係\n"
        "- **赤線（負の因果）**: スキルAを習得すると、スキルBの習得が抑制される関係（競合・代替関係など）\n\n"
        "デフォルトでは正の因果関係のみを表示します。"
    )

    st.warning(
        "⚠️ **パフォーマンスに関する注意**\n\n"
        "グラフのノード数やエッジ数が多いと、ブラウザが重くなったりクラッシュする可能性があります。\n\n"
        "**推奨設定**: 表示ノード数 10-20個、表示閾値 0.3以上から開始してください。"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        display_mode = st.selectbox(
            "表示モード",
            ["全体（主要ノード）", "全体（全ノード）"],
            help="全ノード表示は非常に重くなります。主要ノードモードを推奨します。",
            key="global_display_mode"
        )

    with col2:
        threshold = st.slider(
            "表示閾値（高いほど軽量）",
            0.05, 1.0, 0.3, 0.01,
            key="global_threshold",
            help="この値以上の因果係数を持つエッジのみ表示。高い値ほど表示されるエッジが少なくなり軽量になります。"
        )

    with col3:
        top_n = st.slider(
            "表示ノード数",
            5, 100, 20, 5,
            key="global_top_n",
            help="次数中心性が高い上位Nノードを表示。少ない数から始めることを推奨します。"
        ) if display_mode == "全体（主要ノード）" else 1000


    # 負の因果関係の表示オプション
    show_negative = st.checkbox(
        "負の因果関係も表示する（赤線）",
        value=False,
        key="global_show_negative",
        help="チェックを入れると、負の因果関係（抑制関係）も表示されます。グラフが複雑になる可能性があります。"
    )

    # 自動更新モードのチェックボックス
    auto_update = st.checkbox(
        "設定変更時に自動更新",
        value=False,
        help="チェックを入れると、設定を変更するたびに自動的にグラフを再描画します"
    )

    # 現在の設定
    current_settings = {
        'threshold': threshold,
        'top_n': top_n,
        'show_negative': show_negative,
        'display_mode': display_mode
    }

    # 前回の設定と比較
    settings_changed = False
    if 'global_graph_settings' in st.session_state:
        settings_changed = st.session_state.global_graph_settings != current_settings

    # 描画ボタンまたは自動更新
    should_draw = st.button("🎨 インタラクティブグラフを描画", type="primary")

    # 自動更新がONで設定が変更された場合
    if auto_update and settings_changed and 'global_graph_html' in st.session_state:
        should_draw = True
        st.info("🔄 設定が変更されたため、自動的に再描画します...")

    if should_draw:
        with st.spinner("グラフを生成中..."):
            try:
                adj_matrix = recommender.learner.get_adjacency_matrix()
                visualizer = CausalGraphVisualizer(adj_matrix)

                html_path = visualizer.visualize_interactive(
                    output_path="causal_graph_interactive.html",
                    threshold=threshold,
                    top_n=top_n,
                    show_negative=show_negative,
                    height="800px",
                    width="100%"
                )

                # HTMLファイルを読み込んで保存
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # session_stateに保存
                st.session_state.global_graph_html = html_content
                st.session_state.global_graph_settings = current_settings.copy()

                st.success(f"✅ {top_n}個のノード（次数中心性上位）を表示しました")
                st.caption("💡 ノードをドラッグ・ズーム・クリックして操作できます")

            except Exception as e:
                st.error(f"グラフ描画エラー: {e}")
                st.exception(e)

    # 保存されたグラフを表示
    if 'global_graph_html' in st.session_state:
        components.html(st.session_state.global_graph_html, height=820, scrolling=True)

    # フォールバック: 静的グラフ表示
    with st.expander("📊 静的グラフを表示（軽量版）"):
        if st.button("静的グラフを描画"):
            adj_matrix = recommender.learner.get_adjacency_matrix()
            visualizer = CausalGraphVisualizer(adj_matrix)

            dot = visualizer.visualize(threshold=threshold)
            st.graphviz_chart(dot)
