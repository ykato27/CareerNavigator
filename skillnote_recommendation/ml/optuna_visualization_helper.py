"""
Optuna可視化ヘルパー

チューニング結果を可視化するための統合機能。
"""

import plotly.graph_objects as go
from typing import Optional, List

try:
    import optuna
    import optuna.visualization as vis
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def generate_optuna_visualizations(study: 'optuna.Study', params_to_plot: Optional[List[str]] = None):
    """
    Optunaチューニング結果の可視化グラフを生成

    Args:
        study: Optunaのstudyオブジェクト
        params_to_plot: 可視化するパラメータのリスト（Noneの場合は全パラメータ）

    Returns:
        Dict[str, go.Figure]: 可視化グラフの辞書
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optunaがインストールされていません")

    visualizations = {}

    # 1. 最適化履歴
    try:
        fig_history = vis.plot_optimization_history(study)
        fig_history.update_layout(
            title="チューニング最適化履歴",
            xaxis_title="Trial番号",
            yaxis_title="目的関数値（再構成誤差）",
            height=400
        )
        visualizations['optimization_history'] = fig_history
    except Exception as e:
        print(f"⚠️ 最適化履歴の生成に失敗: {e}")

    # 2. パラメータ重要度
    try:
        fig_importance = vis.plot_param_importances(study)
        fig_importance.update_layout(
            title="ハイパーパラメータ重要度",
            xaxis_title="重要度",
            yaxis_title="パラメータ",
            height=400
        )
        visualizations['param_importances'] = fig_importance
    except Exception as e:
        print(f"⚠️ パラメータ重要度の生成に失敗: {e}")

    # 3. パラレル座標プロット
    try:
        fig_parallel = vis.plot_parallel_coordinate(study, params=params_to_plot)
        fig_parallel.update_layout(
            title="パラレル座標プロット（パラメータ間の関係）",
            height=500
        )
        visualizations['parallel_coordinate'] = fig_parallel
    except Exception as e:
        print(f"⚠️ パラレル座標プロットの生成に失敗: {e}")

    # 4. 等高線図（2パラメータの関係）
    if params_to_plot and len(params_to_plot) >= 2:
        try:
            fig_contour = vis.plot_contour(study, params=params_to_plot[:2])
            fig_contour.update_layout(
                title=f"等高線図（{params_to_plot[0]} vs {params_to_plot[1]}）",
                height=500
            )
            visualizations['contour'] = fig_contour
        except Exception as e:
            print(f"⚠️ 等高線図の生成に失敗: {e}")
    else:
        # デフォルト: n_components と alpha_W
        try:
            fig_contour = vis.plot_contour(study, params=['n_components', 'alpha_W'])
            fig_contour.update_layout(
                title="等高線図（n_components vs alpha_W）",
                height=500
            )
            visualizations['contour'] = fig_contour
        except Exception as e:
            print(f"⚠️ 等高線図の生成に失敗: {e}")

    # 5. スライスプロット（各パラメータの影響）
    try:
        fig_slice = vis.plot_slice(study, params=params_to_plot)
        fig_slice.update_layout(
            title="スライスプロット（各パラメータの影響）",
            height=600
        )
        visualizations['slice'] = fig_slice
    except Exception as e:
        print(f"⚠️ スライスプロットの生成に失敗: {e}")

    # 6. EDF（経験分布関数）
    try:
        fig_edf = vis.plot_edf(study)
        fig_edf.update_layout(
            title="経験分布関数（EDF）",
            xaxis_title="目的関数値",
            yaxis_title="確率",
            height=400
        )
        visualizations['edf'] = fig_edf
    except Exception as e:
        print(f"⚠️ EDFの生成に失敗: {e}")

    return visualizations


def get_best_trials_summary(study: 'optuna.Study', top_n: int = 10):
    """
    上位N個のtrialのサマリーを取得

    Args:
        study: Optunaのstudyオブジェクト
        top_n: 上位何個まで取得するか

    Returns:
        pd.DataFrame: 上位trialの詳細
    """
    import pandas as pd

    trials = study.trials
    trials_sorted = sorted(trials, key=lambda t: t.value if t.value is not None else float('inf'))

    summary_data = []
    for i, trial in enumerate(trials_sorted[:top_n]):
        row = {
            'Rank': i + 1,
            'Trial': trial.number,
            'Value': trial.value,
        }
        row.update(trial.params)

        # ユーザー属性も追加
        if trial.user_attrs:
            row.update({f"attr_{k}": v for k, v in trial.user_attrs.items() if not isinstance(v, list)})

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def get_pruned_trials_count(study: 'optuna.Study') -> dict:
    """
    Pruningされたtrialの統計を取得

    Args:
        study: Optunaのstudyオブジェクト

    Returns:
        dict: 統計情報
    """
    total_trials = len(study.trials)
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    return {
        'total': total_trials,
        'complete': complete_trials,
        'pruned': pruned_trials,
        'failed': failed_trials,
        'pruning_rate': pruned_trials / total_trials * 100 if total_trials > 0 else 0,
    }


def plot_training_history(training_history: dict, title: str = "学習履歴"):
    """
    Early Stoppingの学習履歴をプロット

    Args:
        training_history: MatrixFactorizationModelの学習履歴
        title: グラフタイトル

    Returns:
        go.Figure: Plotlyグラフ
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # 訓練誤差
    if 'train_errors' in training_history:
        fig.add_trace(go.Scatter(
            y=training_history['train_errors'],
            mode='lines',
            name='訓練誤差',
            line=dict(color='blue', width=2)
        ))

    # 検証誤差
    if 'val_errors' in training_history:
        fig.add_trace(go.Scatter(
            y=training_history['val_errors'],
            mode='lines',
            name='検証誤差',
            line=dict(color='red', width=2)
        ))

    # 汎化ギャップ
    if 'generalization_gaps' in training_history:
        fig.add_trace(go.Scatter(
            y=training_history['generalization_gaps'],
            mode='lines',
            name='汎化ギャップ',
            line=dict(color='green', width=2, dash='dash'),
            yaxis='y2'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="評価ステップ（10イテレーションごと）",
        yaxis_title="再構成誤差",
        yaxis2=dict(
            title="汎化ギャップ",
            overlaying='y',
            side='right'
        ),
        height=400,
        showlegend=True,
        hovermode='x unified'
    )

    return fig
