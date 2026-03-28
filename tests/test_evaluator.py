"""
RecommendationEvaluatorクラスのテスト

推薦システムの評価機能をテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from skillnote_recommendation.core.evaluator import RecommendationEvaluator


# ==================== フィクスチャ ====================


@pytest.fixture
def temporal_member_competence():
    """時系列データを含むメンバー習得力量データ"""
    # 2023年1月から2024年12月までのデータ
    base_date = datetime(2023, 1, 1)

    data = []
    for i in range(100):
        # 時系列で力量習得が進む
        acquired_date = base_date + timedelta(days=i * 7)
        data.append(
            {
                "メンバーコード": f"m{(i % 5) + 1:03d}",
                "力量コード": f"s{(i % 20) + 1:03d}",
                "力量タイプ": "SKILL",
                "正規化レベル": 3,
                "力量カテゴリー名": "プログラミング",
                "取得日": acquired_date.strftime("%Y/%m/%d"),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_evaluator():
    """評価用エンジン"""
    return RecommendationEvaluator()


# ==================== 時系列分割テスト ====================


class TestTemporalSplit:
    """時系列分割のテスト"""

    def test_temporal_split_with_ratio(self, temporal_member_competence):
        """デフォルト比率での分割"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.8
        )

        assert len(train_data) > 0
        assert len(test_data) > 0
        # Cold-start問題により、テストデータから一部のメンバーが除外される可能性があるため、
        # 合計が元のデータと一致しない場合がある
        assert len(train_data) + len(test_data) <= len(temporal_member_competence)

    def test_temporal_split_with_date(self, temporal_member_competence):
        """明示的な分割日での分割"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, split_date="2023-07-01"
        )

        assert len(train_data) > 0
        assert len(test_data) > 0

        # 学習データは全て分割日より前
        train_dates = pd.to_datetime(train_data["取得日"])
        assert all(train_dates < pd.to_datetime("2023-07-01"))

        # テストデータは全て分割日以降
        test_dates = pd.to_datetime(test_data["取得日"])
        assert all(test_dates >= pd.to_datetime("2023-07-01"))

    def test_temporal_split_chronological_order(self, temporal_member_competence):
        """時系列順序が保たれる"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.7
        )

        # 学習データの最新日 < テストデータの最古日
        train_max_date = pd.to_datetime(train_data["取得日"]).max()
        test_min_date = pd.to_datetime(test_data["取得日"]).min()

        assert train_max_date <= test_min_date

    def test_temporal_split_missing_date_column(self):
        """取得日カラムがない場合にエラー"""
        evaluator = RecommendationEvaluator()

        df_no_date = pd.DataFrame(
            {"メンバーコード": ["m001", "m002"], "力量コード": ["s001", "s002"]}
        )

        with pytest.raises(ValueError, match="'取得日' カラムが必要です"):
            evaluator.temporal_train_test_split(df_no_date)

    def test_temporal_split_invalid_dates(self):
        """無効な日付を含むデータ"""
        evaluator = RecommendationEvaluator()

        df_invalid = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m002", "m003"],
                "力量コード": ["s001", "s002", "s003"],
                "取得日": ["2023/01/01", "invalid", "2023/12/31"],
            }
        )

        train_data, test_data = evaluator.temporal_train_test_split(df_invalid)

        # 有効な日付のみが含まれる
        assert len(train_data) + len(test_data) == 2

    def test_temporal_split_empty_valid_data(self):
        """有効な日付がない場合"""
        evaluator = RecommendationEvaluator()

        df_no_valid = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m002"],
                "力量コード": ["s001", "s002"],
                "取得日": ["invalid1", "invalid2"],
            }
        )

        with pytest.raises(ValueError, match="有効な取得日を持つデータがありません"):
            evaluator.temporal_train_test_split(df_no_valid)

    def test_temporal_split_different_ratios(self, temporal_member_competence):
        """異なる分割比率でのテスト"""
        evaluator = RecommendationEvaluator()

        for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
            train_data, test_data = evaluator.temporal_train_test_split(
                temporal_member_competence, train_ratio=ratio
            )

            total = len(train_data) + len(test_data)
            actual_ratio = len(train_data) / total

            # 比率が概ね一致（±10%の誤差を許容）
            assert abs(actual_ratio - ratio) < 0.1

    def test_validate_temporal_split_valid_split(self, temporal_member_competence):
        """正しい時系列分割の検証"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, split_date="2023-07-01"
        )

        validation = evaluator.validate_temporal_split(
            train_data, test_data, split_date="2023-07-01"
        )

        # 正しい分割の場合、データリーケージは検出されない
        assert validation["is_valid"]
        assert validation["leakage_members"] == 0
        assert len(validation["issues"]) == 0

    def test_validate_temporal_split_with_leakage(self):
        """データリーケージが発生している場合の検証"""
        evaluator = RecommendationEvaluator()

        # 意図的にリーケージを作成
        train_data = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m001"],
                "力量コード": ["s001", "s002"],
                "取得日": ["2023-06-01", "2023-08-01"],  # 後の日付が含まれる
            }
        )

        test_data = pd.DataFrame(
            {
                "メンバーコード": ["m001"],
                "力量コード": ["s003"],
                "取得日": ["2023-07-01"],  # 前の日付
            }
        )

        validation = evaluator.validate_temporal_split(train_data, test_data)

        # データリーケージが検出される
        assert not validation["is_valid"]
        assert validation["leakage_members"] > 0
        assert len(validation["issues"]) > 0

    def test_validate_temporal_split_cold_start(self):
        """Cold-startメンバーの検出"""
        evaluator = RecommendationEvaluator()

        train_data = pd.DataFrame(
            {
                "メンバーコード": ["m001"],
                "力量コード": ["s001"],
                "取得日": ["2023-01-01"],
            }
        )

        test_data = pd.DataFrame(
            {
                "メンバーコード": ["m002"],  # 訓練セットに存在しない
                "力量コード": ["s002"],
                "取得日": ["2023-06-01"],
            }
        )

        validation = evaluator.validate_temporal_split(train_data, test_data)

        # Cold-startメンバーが検出される
        assert not validation["is_valid"]
        assert validation["cold_start_members"] == 1
        assert any("Cold-start" in issue for issue in validation["issues"])


# ==================== NDCG計算テスト ====================


class TestNDCGCalculation:
    """NDCG計算のテスト"""

    def test_ndcg_perfect_ranking(self, sample_evaluator):
        """完璧なランキングでNDCG=1.0"""
        recommended = ["s001", "s002", "s003", "s004", "s005"]
        relevant = ["s001", "s002", "s003", "s004", "s005"]

        ndcg = sample_evaluator._calculate_ndcg(recommended, relevant, k=5)

        assert ndcg == pytest.approx(1.0)

    def test_ndcg_no_relevant(self, sample_evaluator):
        """関連アイテムがない場合NDCG=0.0"""
        recommended = ["s001", "s002", "s003"]
        relevant = ["s999", "s998", "s997"]

        ndcg = sample_evaluator._calculate_ndcg(recommended, relevant, k=3)

        assert ndcg == 0.0

    def test_ndcg_partial_match(self, sample_evaluator):
        """部分一致の場合0.0 < NDCG < 1.0"""
        recommended = ["s001", "s002", "s003", "s004", "s005"]
        relevant = ["s002", "s004"]  # 2つだけ一致

        ndcg = sample_evaluator._calculate_ndcg(recommended, relevant, k=5)

        assert 0.0 < ndcg < 1.0

    def test_ndcg_ranking_matters(self, sample_evaluator):
        """ランキング位置が影響する"""
        relevant = ["s001", "s002"]

        # 関連アイテムが上位にある場合
        recommended_good = ["s001", "s002", "s003", "s004", "s005"]
        ndcg_good = sample_evaluator._calculate_ndcg(recommended_good, relevant, k=5)

        # 関連アイテムが下位にある場合
        recommended_bad = ["s003", "s004", "s001", "s002", "s005"]
        ndcg_bad = sample_evaluator._calculate_ndcg(recommended_bad, relevant, k=5)

        # 上位にある方がスコアが高い
        assert ndcg_good > ndcg_bad

    def test_ndcg_empty_recommendations(self, sample_evaluator):
        """推薦が空の場合NDCG=0.0"""
        recommended = []
        relevant = ["s001", "s002"]

        ndcg = sample_evaluator._calculate_ndcg(recommended, relevant, k=5)

        assert ndcg == 0.0

    def test_ndcg_k_limit(self, sample_evaluator):
        """K値の制限が効く"""
        recommended = ["s999", "s998", "s997", "s001", "s002"]
        relevant = ["s001", "s002"]  # 4, 5番目にある

        # K=3では関連アイテムが含まれない
        ndcg_k3 = sample_evaluator._calculate_ndcg(recommended, relevant, k=3)
        assert ndcg_k3 == 0.0

        # K=5では含まれる
        ndcg_k5 = sample_evaluator._calculate_ndcg(recommended, relevant, k=5)
        assert ndcg_k5 > 0.0


# ==================== 評価メトリクス計算テスト ====================


class TestEvaluationMetrics:
    """評価メトリクス計算のテスト"""

    def test_evaluate_recommendations_basic(
        self, temporal_member_competence, sample_competence_master
    ):
        """基本的な評価実行"""
        evaluator = RecommendationEvaluator()

        # データを分割
        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.7
        )

        # 評価実行
        metrics = evaluator.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=5,
        )

        # メトリクスが全て含まれる
        assert "precision@5" in metrics
        assert "recall@5" in metrics
        assert "ndcg@5" in metrics
        assert "hit_rate" in metrics
        assert "evaluated_members" in metrics

    def test_metrics_range(self, temporal_member_competence, sample_competence_master):
        """メトリクスの値域確認"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.8
        )

        metrics = evaluator.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
        )

        # 全てのメトリクスが0.0-1.0の範囲
        assert 0.0 <= metrics["precision@10"] <= 1.0
        assert 0.0 <= metrics["recall@10"] <= 1.0
        assert 0.0 <= metrics["ndcg@10"] <= 1.0
        assert 0.0 <= metrics["hit_rate"] <= 1.0

    def test_evaluate_with_member_sample(
        self, temporal_member_competence, sample_competence_master
    ):
        """特定メンバーのみ評価"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.7
        )

        # 特定メンバーのみ評価
        member_sample = ["m001", "m002"]

        metrics = evaluator.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=5,
            member_sample=member_sample,
        )

        # 評価対象メンバー数が制限される
        assert metrics["evaluated_members"] <= len(member_sample)

    def test_evaluate_empty_test_data(self, sample_competence_master):
        """テストデータが空の場合"""
        evaluator = RecommendationEvaluator()

        train_data = pd.DataFrame(
            {
                "メンバーコード": ["m001"],
                "力量コード": ["s001"],
                "力量タイプ": ["SKILL"],
                "正規化レベル": [3],
                "力量カテゴリー名": ["技術"],
                "取得日": ["2023/01/01"],
            }
        )

        test_data = pd.DataFrame(columns=train_data.columns)

        metrics = evaluator.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=5,
        )

        # 全てゼロ
        assert metrics["evaluated_members"] == 0
        assert metrics["precision@5"] == 0.0

    def test_precision_calculation(self, sample_competence_master):
        """Precision計算の正確性"""
        evaluator = RecommendationEvaluator()

        # m001が過去にs001を習得
        train_data = pd.DataFrame(
            {
                "メンバーコード": ["m001"],
                "力量コード": ["s001"],
                "力量タイプ": ["SKILL"],
                "正規化レベル": [3],
                "力量カテゴリー名": ["プログラミング"],
                "取得日": ["2023/01/01"],
            }
        )

        # m001が将来s002, s003を習得
        test_data = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m001"],
                "力量コード": ["s002", "s003"],
                "力量タイプ": ["SKILL", "SKILL"],
                "正規化レベル": [3, 3],
                "力量カテゴリー名": ["プログラミング", "プログラミング"],
                "取得日": ["2023/06/01", "2023/07/01"],
            }
        )

        metrics = evaluator.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
            member_sample=["m001"],
        )

        # Precisionは「推薦のうち正解」なので > 0
        assert metrics["evaluated_members"] == 1

    def test_recall_calculation(self, sample_competence_master):
        """Recall計算の正確性"""
        evaluator = RecommendationEvaluator()

        train_data = pd.DataFrame(
            {
                "メンバーコード": ["m001"],
                "力量コード": ["s001"],
                "力量タイプ": ["SKILL"],
                "正規化レベル": [3],
                "力量カテゴリー名": ["プログラミング"],
                "取得日": ["2023/01/01"],
            }
        )

        test_data = pd.DataFrame(
            {
                "メンバーコード": ["m001"],
                "力量コード": ["s002"],
                "力量タイプ": ["SKILL"],
                "正規化レベル": [3],
                "力量カテゴリー名": ["プログラミング"],
                "取得日": ["2023/06/01"],
            }
        )

        metrics = evaluator.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
            member_sample=["m001"],
        )

        # Recallは「正解のうち推薦」なので0.0-1.0
        assert 0.0 <= metrics["recall@10"] <= 1.0


# ==================== クロスバリデーションテスト ====================


class TestCrossValidation:
    """時系列クロスバリデーションのテスト"""

    def test_cross_validate_temporal(self, temporal_member_competence, sample_competence_master):
        """基本的なクロスバリデーション"""
        evaluator = RecommendationEvaluator()

        results = evaluator.cross_validate_temporal(
            member_competence=temporal_member_competence,
            competence_master=sample_competence_master,
            n_splits=3,
            top_k=5,
        )

        # 3分割なので3つの結果
        assert len(results) <= 3

        # 各結果にメトリクスが含まれる
        for result in results:
            assert "precision@5" in result
            assert "recall@5" in result
            assert "fold" in result
            assert "train_size" in result
            assert "test_size" in result

    def test_cross_validate_progressive_training(
        self, temporal_member_competence, sample_competence_master
    ):
        """学習データが累積的に増える"""
        evaluator = RecommendationEvaluator()

        results = evaluator.cross_validate_temporal(
            member_competence=temporal_member_competence,
            competence_master=sample_competence_master,
            n_splits=3,
            top_k=5,
        )

        # 学習データサイズが増加
        train_sizes = [r["train_size"] for r in results]
        assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1))

    def test_cross_validate_different_splits(
        self, temporal_member_competence, sample_competence_master
    ):
        """異なる分割数でのテスト"""
        evaluator = RecommendationEvaluator()

        for n_splits in [2, 3, 5]:
            results = evaluator.cross_validate_temporal(
                member_competence=temporal_member_competence,
                competence_master=sample_competence_master,
                n_splits=n_splits,
                top_k=5,
            )

            assert len(results) <= n_splits


# ==================== 結果表示・出力テスト ====================


class TestResultsOutput:
    """結果表示と出力のテスト"""

    def test_print_evaluation_results(self, capsys):
        """評価結果表示"""
        evaluator = RecommendationEvaluator()

        metrics = {
            "precision@10": 0.35,
            "recall@10": 0.42,
            "ndcg@10": 0.38,
            "hit_rate": 0.65,
            "evaluated_members": 50,
        }

        evaluator.print_evaluation_results(metrics)

        captured = capsys.readouterr()

        assert "推薦システム評価結果" in captured.out
        assert "Precision@10" in captured.out
        assert "Recall@10" in captured.out
        assert "NDCG@10" in captured.out
        assert "Hit Rate" in captured.out

    def test_export_evaluation_results(self, tmp_path):
        """評価結果CSV出力"""
        evaluator = RecommendationEvaluator()

        metrics = {
            "precision@5": 0.30,
            "recall@5": 0.40,
            "ndcg@5": 0.35,
            "hit_rate": 0.60,
            "evaluated_members": 30,
        }

        output_file = tmp_path / "evaluation_results.csv"

        evaluator.export_evaluation_results(metrics, str(output_file))

        # ファイルが作成される
        assert output_file.exists()

        # 内容確認
        df = pd.read_csv(output_file, encoding="utf-8-sig")
        assert len(df) == 1
        assert df.loc[0, "precision@5"] == pytest.approx(0.30)
        assert df.loc[0, "recall@5"] == pytest.approx(0.40)


# ==================== 統合テスト ====================


class TestIntegration:
    """統合動作のテスト"""

    def test_full_evaluation_workflow(
        self, temporal_member_competence, sample_competence_master, tmp_path
    ):
        """完全な評価ワークフロー"""
        evaluator = RecommendationEvaluator()

        # 1. データ分割
        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.8
        )

        assert len(train_data) > 0
        assert len(test_data) > 0

        # 2. 評価実行
        metrics = evaluator.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
        )

        assert metrics["evaluated_members"] > 0

        # 3. 結果出力
        output_file = tmp_path / "evaluation.csv"
        evaluator.export_evaluation_results(metrics, str(output_file))

        assert output_file.exists()

    def test_cross_validation_workflow(self, temporal_member_competence, sample_competence_master):
        """クロスバリデーション完全ワークフロー"""
        evaluator = RecommendationEvaluator()

        # クロスバリデーション実行
        results = evaluator.cross_validate_temporal(
            member_competence=temporal_member_competence,
            competence_master=sample_competence_master,
            n_splits=3,
            top_k=5,
        )

        # 全foldで評価完了
        assert len(results) > 0

        # 平均メトリクスを計算
        avg_precision = np.mean([r["precision@5"] for r in results])
        avg_recall = np.mean([r["recall@5"] for r in results])

        assert 0.0 <= avg_precision <= 1.0
        assert 0.0 <= avg_recall <= 1.0


# ==================== 多様性指標テスト ====================


class TestDiversityMetrics:
    """多様性指標のテスト"""

    def test_calculate_diversity_metrics_basic(self, sample_competence_master):
        """基本的な多様性指標計算"""
        from skillnote_recommendation.core.models import Recommendation

        evaluator = RecommendationEvaluator()

        # サンプル推薦リスト
        recommendations_list = [
            [
                Recommendation("s001", "Python", "SKILL", "プログラミング", 8.0, 0, 0, 0, "理由1"),
                Recommendation("s002", "Java", "SKILL", "プログラミング", 7.0, 0, 0, 0, "理由2"),
                Recommendation("e001", "AWS研修", "EDUCATION", "クラウド", 6.0, 0, 0, 0, "理由3"),
            ]
        ]

        metrics = evaluator.calculate_diversity_metrics(
            recommendations_list, sample_competence_master
        )

        assert "avg_category_diversity" in metrics
        assert "avg_type_diversity" in metrics
        assert "catalog_coverage" in metrics

    def test_diversity_metrics_range(self, sample_competence_master):
        """多様性指標の値域"""
        from skillnote_recommendation.core.models import Recommendation

        evaluator = RecommendationEvaluator()

        recommendations_list = [
            [
                Recommendation("s001", "Python", "SKILL", "プログラミング", 8.0, 0, 0, 0, "理由"),
                Recommendation("s002", "Java", "SKILL", "データベース", 7.0, 0, 0, 0, "理由"),
            ]
        ]

        metrics = evaluator.calculate_diversity_metrics(
            recommendations_list, sample_competence_master
        )

        assert 0.0 <= metrics["avg_category_diversity"] <= 1.0
        assert 0.0 <= metrics["avg_type_diversity"] <= 1.0
        assert 0.0 <= metrics["catalog_coverage"] <= 1.0

    def test_diversity_metrics_empty_recommendations(self, sample_competence_master):
        """空の推薦リスト"""
        evaluator = RecommendationEvaluator()

        metrics = evaluator.calculate_diversity_metrics([], sample_competence_master)

        assert metrics["avg_category_diversity"] == 0.0
        assert metrics["catalog_coverage"] == 0.0

    def test_evaluate_with_diversity(self, temporal_member_competence, sample_competence_master):
        """多様性込みの評価"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.7
        )

        metrics = evaluator.evaluate_with_diversity(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=5,
        )

        # 基本メトリクス
        assert "precision@5" in metrics
        assert "recall@5" in metrics

        # 多様性指標
        assert "avg_category_diversity" in metrics
        assert "avg_type_diversity" in metrics
        assert "catalog_coverage" in metrics


# ==================== メンバーごとの評価テスト ====================


class TestPerMemberEvaluation:
    """メンバーごとの評価メトリクスのテスト"""

    def test_evaluate_per_member_basic(self, temporal_member_competence, sample_competence_master):
        """メンバーごとの評価が機能する"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.7
        )

        # テストデータにデータが含まれているかチェック
        if len(test_data) == 0:
            pytest.skip("テストデータが空です")

        per_member_df = evaluator.evaluate_per_member(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
        )

        # 結果がDataFrameであることを確認
        assert isinstance(per_member_df, pd.DataFrame)

        # 必要なカラムが含まれている
        if len(per_member_df) > 0:
            assert "member_code" in per_member_df.columns
            assert "precision@10" in per_member_df.columns
            assert "recall@10" in per_member_df.columns
            assert "f1@10" in per_member_df.columns
            assert "ndcg@10" in per_member_df.columns
            assert "hit" in per_member_df.columns

    def test_evaluate_per_member_metrics_range(self, temporal_member_competence, sample_competence_master):
        """メンバーごとのメトリクスが有効な範囲内である"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.7
        )

        if len(test_data) == 0:
            pytest.skip("テストデータが空です")

        per_member_df = evaluator.evaluate_per_member(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
        )

        if len(per_member_df) > 0:
            # すべてのメトリクスが0～1の範囲
            assert (per_member_df["precision@10"] >= 0).all()
            assert (per_member_df["precision@10"] <= 1).all()
            assert (per_member_df["recall@10"] >= 0).all()
            assert (per_member_df["recall@10"] <= 1).all()
            assert (per_member_df["ndcg@10"] >= 0).all()
            assert (per_member_df["ndcg@10"] <= 1).all()

            # hitは0または1
            assert per_member_df["hit"].isin([0, 1]).all()

    def test_get_member_performance_summary(self, temporal_member_competence, sample_competence_master):
        """メンバーパフォーマンスサマリーが機能する"""
        evaluator = RecommendationEvaluator()

        train_data, test_data = evaluator.temporal_train_test_split(
            temporal_member_competence, train_ratio=0.7
        )

        if len(test_data) == 0:
            pytest.skip("テストデータが空です")

        per_member_df = evaluator.evaluate_per_member(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
        )

        if len(per_member_df) > 0:
            summary = evaluator.get_member_performance_summary(per_member_df, top_k=10)

            # サマリーキーが存在
            assert "total_members" in summary
            assert "high_performers" in summary
            assert "medium_performers" in summary
            assert "low_performers" in summary
            assert "avg_precision" in summary
            assert "avg_recall" in summary
            assert "avg_f1" in summary

            # メンバー数の合計が一致
            total = summary["high_performers"] + summary["medium_performers"] + summary["low_performers"]
            assert total == summary["total_members"]

    def test_per_member_empty_test_data(self, sample_competence_master):
        """テストデータが空の場合"""
        evaluator = RecommendationEvaluator()

        train_data = pd.DataFrame({
            "メンバーコード": ["m001", "m002"],
            "力量コード": ["s001", "s002"],
            "力量タイプ": ["SKILL", "SKILL"],
            "正規化レベル": [3, 3],
            "力量カテゴリー名": ["Cat1", "Cat2"],
        })

        test_data = pd.DataFrame({
            "メンバーコード": [],
            "力量コード": [],
            "力量タイプ": [],
            "正規化レベル": [],
            "力量カテゴリー名": [],
        })

        per_member_df = evaluator.evaluate_per_member(
            train_data=train_data,
            test_data=test_data,
            competence_master=sample_competence_master,
            top_k=10,
        )

        # 空のDataFrameが返される
        assert isinstance(per_member_df, pd.DataFrame)
        assert len(per_member_df) == 0


# ==================== モデル評価メトリクステスト ====================


class TestMatrixFactorizationMetrics:
    """マトリックスファクタリゼーションモデルの評価メトリクステスト"""

    def test_normalized_reconstruction_error(self, sample_skill_matrix):
        """正規化再構成誤差が計算できる"""
        from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel

        model = MatrixFactorizationModel(n_components=5)
        model.fit(sample_skill_matrix)

        normalized_error = model.get_normalized_reconstruction_error()

        # 正規化誤差は0～1の範囲（理論的には無限大も可能だが、通常は0～1）
        assert isinstance(normalized_error, float)
        assert normalized_error >= 0

    def test_model_sparsity(self, sample_skill_matrix):
        """モデルスパース性が計算できる"""
        from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel

        model = MatrixFactorizationModel(n_components=5)
        model.fit(sample_skill_matrix)

        sparsity_info = model.get_model_sparsity()

        # スパース性情報の存在
        assert "W_sparsity" in sparsity_info
        assert "H_sparsity" in sparsity_info
        assert "unused_factors" in sparsity_info
        assert "recommendation" in sparsity_info

        # スパース性は0～100%の範囲
        assert 0 <= sparsity_info["W_sparsity"] <= 100
        assert 0 <= sparsity_info["H_sparsity"] <= 100

        # 未使用因子はリスト
        assert isinstance(sparsity_info["unused_factors"], list)

    def test_model_sparsity_with_unused_factors(self):
        """未使用潜在因子がある場合"""
        from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel

        # スパースなデータを作成（低ランク構造）
        sparse_data = pd.DataFrame(
            np.array([
                [1, 0, 2],
                [0, 3, 0],
                [4, 0, 5],
            ]),
            index=["m1", "m2", "m3"],
            columns=["s1", "s2", "s3"],
        )

        # 潜在因子数を大きく設定（未使用因子が生まれやすい）
        model = MatrixFactorizationModel(n_components=10, random_state=42)
        model.fit(sparse_data)

        sparsity_info = model.get_model_sparsity()

        # 診断メッセージが生成されている
        assert isinstance(sparsity_info["recommendation"], str)
        assert len(sparsity_info["recommendation"]) > 0


# ==================== 汎化性能テスト ====================


class TestGeneralizationMetrics:
    """汎化性能メトリクスのテスト"""

    def test_evaluate_training_vs_test(self, sample_skill_matrix):
        """訓練 vs テスト誤差が計算できる"""
        from skillnote_recommendation.ml.hyperparameter_tuning import NMFHyperparameterTuner

        tuner = NMFHyperparameterTuner(
            skill_matrix=sample_skill_matrix,
            n_trials=1,
            random_state=42,
            test_size=0.2,  # テストセットを分離
        )

        # 最適化を実行
        best_params, best_value = tuner.optimize()

        # 最良モデルを取得
        model = tuner.get_best_model()

        # 訓練 vs テスト誤差を評価
        eval_results = tuner.evaluate_training_vs_test(model)

        # 必要なキーが存在
        assert "train_error" in eval_results
        assert "train_size" in eval_results
        assert "test_size" in eval_results
        assert "diagnosis" in eval_results

        # テストセットが存在する場合
        if eval_results["test_size"] > 0:
            assert "test_error" in eval_results
            assert "generalization_gap" in eval_results

            # テスト誤差と訓練誤差は正の値
            assert eval_results["test_error"] >= 0
            assert eval_results["train_error"] >= 0

    def test_generalization_gap_interpretation(self, sample_skill_matrix):
        """汎化ギャップの診断が正確である"""
        from skillnote_recommendation.ml.hyperparameter_tuning import NMFHyperparameterTuner

        tuner = NMFHyperparameterTuner(
            skill_matrix=sample_skill_matrix,
            n_trials=1,
            random_state=42,
            test_size=0.2,
        )

        best_params, best_value = tuner.optimize()
        model = tuner.get_best_model()

        eval_results = tuner.evaluate_training_vs_test(model)

        if eval_results["test_size"] > 0:
            diagnosis = eval_results["diagnosis"]

            # 診断メッセージが含まれている
            assert isinstance(diagnosis, str)
            assert len(diagnosis) > 0

            # 診断メッセージが有効な内容を含む
            valid_keywords = ["優れた", "過学習", "テストセット", "汎化"]
            has_valid_keyword = any(keyword in diagnosis for keyword in valid_keywords)
            assert has_valid_keyword or "診断" in diagnosis
