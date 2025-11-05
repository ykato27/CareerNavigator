"""
推薦システム評価スクリプト

使用例:
    python skillnote_recommendation/scripts/evaluate_recommendations.py

機能:
    - Hold-out評価
    - 交差検証
    - Leave-One-Out評価
    - 多様性・カバレッジ評価
"""

import pandas as pd
import logging
from pathlib import Path

from skillnote_recommendation.core.config import Config
from skillnote_recommendation.core.evaluator import RecommendationEvaluator
from skillnote_recommendation.ml.ml_evaluation import MLRecommendationEvaluator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """データを読み込み"""
    logger.info("データ読み込み中...")

    # 出力ディレクトリからデータを読み込み
    output_dir = Path(Config.OUTPUT_DIR)

    member_competence = pd.read_csv(
        output_dir / 'member_competence.csv',
        encoding=Config.FILE_ENCODING
    )
    competence_master = pd.read_csv(
        output_dir / 'competence_master.csv',
        encoding=Config.FILE_ENCODING
    )
    member_master = pd.read_csv(
        output_dir / 'members_clean.csv',
        encoding=Config.FILE_ENCODING
    )

    logger.info(f"メンバー習得力量: {len(member_competence)}件")
    logger.info(f"力量マスタ: {len(competence_master)}件")
    logger.info(f"メンバーマスタ: {len(member_master)}件")

    return member_competence, competence_master, member_master


def evaluate_hold_out(
    member_competence: pd.DataFrame,
    competence_master: pd.DataFrame,
    member_master: pd.DataFrame
):
    """
    Hold-out評価を実行

    Args:
        member_competence: メンバー習得力量データ
        competence_master: 力量マスタ
        member_master: メンバーマスタ
    """
    logger.info("\n" + "="*80)
    logger.info("Hold-out評価を開始")
    logger.info("="*80)

    # 評価器を初期化
    ml_evaluator = MLRecommendationEvaluator()

    # 評価実行
    metrics = ml_evaluator.evaluate_with_holdout(
        member_competence=member_competence,
        competence_master=competence_master,
        member_master=member_master,
        train_ratio=Config.EVALUATION_PARAMS['train_ratio'],
        top_k=Config.EVALUATION_PARAMS['top_k'],
        use_temporal_split=Config.EVALUATION_PARAMS['use_temporal_split'],
        random_state=42
    )

    # 結果表示
    ml_evaluator.print_ml_evaluation_results(metrics)

    # 結果をCSVに出力
    if Config.EVALUATION_PARAMS['export_results']:
        output_path = Path(Config.OUTPUT_DIR) / 'evaluation_holdout_results.csv'
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False, encoding=Config.OUTPUT_ENCODING)
        logger.info(f"\n評価結果を保存: {output_path}")

    return metrics


def evaluate_cross_validation(
    member_competence: pd.DataFrame,
    competence_master: pd.DataFrame,
    member_master: pd.DataFrame
):
    """
    交差検証を実行

    Args:
        member_competence: メンバー習得力量データ
        competence_master: 力量マスタ
        member_master: メンバーマスタ
    """
    logger.info("\n" + "="*80)
    logger.info("交差検証を開始")
    logger.info("="*80)

    # 評価器を初期化
    ml_evaluator = MLRecommendationEvaluator()

    # 評価実行
    avg_metrics, fold_metrics = ml_evaluator.evaluate_with_cross_validation(
        member_competence=member_competence,
        competence_master=competence_master,
        member_master=member_master,
        n_folds=Config.EVALUATION_PARAMS['n_folds'],
        top_k=Config.EVALUATION_PARAMS['top_k'],
        use_temporal=Config.EVALUATION_PARAMS['cv_use_temporal'],
        random_state=42
    )

    # 結果表示
    ml_evaluator.print_ml_evaluation_results(avg_metrics, fold_metrics)

    # 結果をCSVに出力
    if Config.EVALUATION_PARAMS['export_results']:
        # 平均メトリクス
        output_path = Path(Config.OUTPUT_DIR) / 'evaluation_cv_average.csv'
        df_avg = pd.DataFrame([avg_metrics])
        df_avg.to_csv(output_path, index=False, encoding=Config.OUTPUT_ENCODING)
        logger.info(f"\n平均メトリクスを保存: {output_path}")

        # 各Foldのメトリクス
        output_path_folds = Path(Config.OUTPUT_DIR) / 'evaluation_cv_folds.csv'
        df_folds = pd.DataFrame(fold_metrics)
        df_folds.to_csv(output_path_folds, index=False, encoding=Config.OUTPUT_ENCODING)
        logger.info(f"Fold別メトリクスを保存: {output_path_folds}")

    return avg_metrics, fold_metrics


def evaluate_leave_one_out(
    member_competence: pd.DataFrame,
    competence_master: pd.DataFrame
):
    """
    Leave-One-Out評価を実行

    Args:
        member_competence: メンバー習得力量データ
        competence_master: 力量マスタ
    """
    logger.info("\n" + "="*80)
    logger.info("Leave-One-Out評価を開始")
    logger.info("="*80)

    # 評価器を初期化
    evaluator = RecommendationEvaluator()

    # 評価実行
    metrics = evaluator.evaluate_leave_one_out(
        member_competence=member_competence,
        competence_master=competence_master,
        top_k=Config.EVALUATION_PARAMS['top_k'],
        max_users=Config.EVALUATION_PARAMS['loo_max_users'],
        random_state=42
    )

    # 結果表示
    evaluator.print_evaluation_results(metrics, detailed=True)

    # 結果をCSVに出力
    if Config.EVALUATION_PARAMS['export_results']:
        output_path = Path(Config.OUTPUT_DIR) / 'evaluation_loo_results.csv'
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False, encoding=Config.OUTPUT_ENCODING)
        logger.info(f"\n評価結果を保存: {output_path}")

    return metrics


def main():
    """メイン処理"""
    try:
        # データ読み込み
        member_competence, competence_master, member_master = load_data()

        # 評価タイプを選択
        logger.info("\n" + "="*80)
        logger.info("評価タイプを選択してください:")
        logger.info("  1. Hold-out評価（デフォルト）")
        logger.info("  2. 交差検証")
        logger.info("  3. Leave-One-Out評価")
        logger.info("  4. 全て実行")
        logger.info("="*80)

        # この例では Hold-out評価をデフォルトで実行
        evaluation_type = 1  # 環境変数やコマンドライン引数から取得することも可能

        if evaluation_type == 1:
            # Hold-out評価
            evaluate_hold_out(member_competence, competence_master, member_master)

        elif evaluation_type == 2:
            # 交差検証
            evaluate_cross_validation(member_competence, competence_master, member_master)

        elif evaluation_type == 3:
            # Leave-One-Out評価
            evaluate_leave_one_out(member_competence, competence_master)

        elif evaluation_type == 4:
            # 全て実行
            logger.info("\n全ての評価を順次実行します...")

            # 1. Hold-out評価
            evaluate_hold_out(member_competence, competence_master, member_master)

            # 2. 交差検証
            evaluate_cross_validation(member_competence, competence_master, member_master)

            # 3. Leave-One-Out評価（サンプリング）
            Config.EVALUATION_PARAMS['loo_max_users'] = 100  # 100ユーザーのみ
            evaluate_leave_one_out(member_competence, competence_master)

        logger.info("\n" + "="*80)
        logger.info("全ての評価が完了しました！")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
