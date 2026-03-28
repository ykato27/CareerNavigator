"""
データ品質モニタリングモジュールのテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from skillnote_recommendation.core.data_quality_monitor import (
    DataQualityMonitor,
    DataQualityIssue,
    DataQualityReport,
    Severity,
)


@pytest.fixture
def sample_member_competence():
    """正常なメンバー習得力量データ"""
    return pd.DataFrame(
        {
            "メンバーコード": ["M001", "M001", "M002", "M002", "M003"],
            "力量コード": ["SKILL001", "SKILL002", "SKILL001", "SKILL003", "SKILL002"],
            "レベル": [3, 4, 2, 5, 3],
            "正規化レベル": [3, 4, 2, 5, 3],
            "取得日": ["2024/01/15", "2024/02/20", "2024/01/10", "2024/03/05", "2024/02/25"],
        }
    )


@pytest.fixture
def sample_competence_master():
    """力量マスタ"""
    return pd.DataFrame(
        {
            "力量コード": ["SKILL001", "SKILL002", "SKILL003"],
            "力量名": ["Python基礎", "Python応用", "データ分析"],
            "力量タイプ": ["SKILL", "SKILL", "SKILL"],
            "力量カテゴリー名": ["プログラミング", "プログラミング", "データサイエンス"],
        }
    )


@pytest.fixture
def monitor():
    """データ品質モニター"""
    return DataQualityMonitor(missing_threshold=0.05, staleness_days=180, max_skills_per_week=3)


@pytest.fixture
def monitor_with_dependencies():
    """スキル依存関係を含むモニター"""
    return DataQualityMonitor(
        missing_threshold=0.05,
        staleness_days=180,
        max_skills_per_week=3,
        skill_dependencies={
            "SKILL002": ["SKILL001"],  # Python応用はPython基礎が前提
            "SKILL003": ["SKILL001"],  # データ分析はPython基礎が前提
        },
    )


class TestDataQualityMonitorInitialization:
    """初期化のテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化"""
        monitor = DataQualityMonitor()
        assert monitor.missing_threshold == 0.05
        assert monitor.staleness_days == 180
        assert monitor.max_skills_per_week == 3
        assert monitor.skill_dependencies == {}

    def test_initialization_with_custom_values(self):
        """カスタム値での初期化"""
        monitor = DataQualityMonitor(
            missing_threshold=0.1,
            staleness_days=90,
            max_skills_per_week=5,
            skill_dependencies={"SKILL002": ["SKILL001"]},
        )
        assert monitor.missing_threshold == 0.1
        assert monitor.staleness_days == 90
        assert monitor.max_skills_per_week == 5
        assert "SKILL002" in monitor.skill_dependencies


class TestCompletenessCheck:
    """完全性チェックのテスト"""

    def test_no_missing_data(self, monitor, sample_member_competence):
        """欠損値がない場合"""
        issues = monitor.check_completeness(sample_member_competence)
        assert len(issues) == 0

    def test_missing_required_column(self, monitor):
        """必須カラムが欠損している場合"""
        df = pd.DataFrame({"力量コード": ["SKILL001", "SKILL002"], "レベル": [3, 4]})
        issues = monitor.check_completeness(df)
        assert len(issues) > 0
        assert issues[0].severity == Severity.CRITICAL
        assert "メンバーコード" in issues[0].title

    def test_high_missing_ratio(self, monitor):
        """欠損率が高い場合"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M002", "M003", "M004", "M005"] * 4,
                "力量コード": ["SKILL001"] * 20,
                "レベル": [3, np.nan, np.nan, 4, np.nan] * 4,  # 60%欠損
            }
        )
        issues = monitor.check_completeness(df)
        assert len(issues) > 0
        # 欠損率が高いため警告が出る
        level_issues = [i for i in issues if "レベル" in i.title]
        assert len(level_issues) > 0
        assert level_issues[0].severity in [Severity.MEDIUM, Severity.HIGH]

    def test_empty_rows(self, monitor):
        """完全に空のレコードがある場合"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", np.nan, "M002"],
                "力量コード": ["SKILL001", np.nan, "SKILL002"],
                "レベル": [3, np.nan, 4],
            }
        )
        issues = monitor.check_completeness(df)
        empty_row_issues = [i for i in issues if "空のレコード" in i.title]
        assert len(empty_row_issues) > 0
        assert empty_row_issues[0].severity == Severity.HIGH


class TestConsistencyCheck:
    """一貫性チェックのテスト"""

    def test_valid_level_range(self, monitor, sample_member_competence):
        """レベル値が正常範囲の場合"""
        issues = monitor.check_consistency(sample_member_competence)
        level_issues = [i for i in issues if "レベル値が範囲外" in i.title]
        assert len(level_issues) == 0

    def test_invalid_level_range(self, monitor):
        """レベル値が範囲外の場合"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M002", "M003"],
                "力量コード": ["SKILL001", "SKILL002", "SKILL003"],
                "レベル": [10, -1, 3],  # 10と-1は範囲外
                "正規化レベル": [10, -1, 3],
            }
        )
        issues = monitor.check_consistency(df)
        level_issues = [i for i in issues if "レベル値が範囲外" in i.title]
        assert len(level_issues) > 0
        assert level_issues[0].severity == Severity.HIGH
        assert level_issues[0].affected_records == 2

    def test_level_zero_with_acquisition_date(self, monitor):
        """レベル0だが取得日が記録されている場合"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M002"],
                "力量コード": ["SKILL001", "SKILL002"],
                "正規化レベル": [0, 3],
                "取得日": ["2024/01/15", "2024/02/20"],  # M001はレベル0だが取得日あり
            }
        )
        issues = monitor.check_consistency(df)
        zero_level_issues = [i for i in issues if "レベル0だが取得日" in i.title]
        assert len(zero_level_issues) > 0
        assert zero_level_issues[0].severity == Severity.LOW

    def test_prerequisite_violations(self, monitor_with_dependencies):
        """前提スキルの依存関係違反"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M001"],
                "力量コード": ["SKILL002", "SKILL003"],  # SKILL001なしでSKILL002とSKILL003
                "取得日": ["2024/01/15", "2024/02/20"],
            }
        )
        issues = monitor_with_dependencies.check_consistency(df)
        prereq_issues = [i for i in issues if "前提スキル" in i.title]
        assert len(prereq_issues) > 0
        assert prereq_issues[0].severity == Severity.HIGH

    def test_temporal_consistency_violation(self, monitor_with_dependencies):
        """時系列の整合性違反（前提スキルより前に上級スキルを習得）"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M001"],
                "力量コード": ["SKILL001", "SKILL002"],
                "取得日": ["2024/02/01", "2024/01/01"],  # SKILL002の方が先
            }
        )
        issues = monitor_with_dependencies.check_consistency(df)
        temporal_issues = [i for i in issues if "時系列" in i.title]
        assert len(temporal_issues) > 0
        assert temporal_issues[0].severity == Severity.MEDIUM


class TestDuplicatesCheck:
    """重複チェックのテスト"""

    def test_no_duplicates(self, monitor, sample_member_competence):
        """重複がない場合"""
        issues = monitor.check_duplicates(sample_member_competence)
        assert len(issues) == 0

    def test_exact_duplicates(self, monitor):
        """完全重複がある場合"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M001", "M002"],
                "力量コード": ["SKILL001", "SKILL001", "SKILL002"],
                "取得日": ["2024/01/15", "2024/01/15", "2024/02/20"],
            }
        )
        issues = monitor.check_duplicates(df)
        duplicate_issues = [i for i in issues if "重複レコード" in i.title]
        assert len(duplicate_issues) > 0
        assert duplicate_issues[0].severity == Severity.MEDIUM
        assert duplicate_issues[0].affected_records == 2

    def test_multiple_acquisition_dates(self, monitor):
        """同じスキルを複数回習得（取得日が異なる）"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M001", "M001"],
                "力量コード": ["SKILL001", "SKILL001", "SKILL002"],
                "取得日": ["2024/01/15", "2024/02/20", "2024/03/01"],
            }
        )
        issues = monitor.check_duplicates(df)
        multi_acq_issues = [i for i in issues if "複数回習得" in i.title]
        assert len(multi_acq_issues) > 0
        assert multi_acq_issues[0].severity == Severity.LOW


class TestTimelinessCheck:
    """適時性チェックのテスト"""

    def test_missing_date_column(self, monitor):
        """取得日カラムがない場合"""
        df = pd.DataFrame(
            {"メンバーコード": ["M001", "M002"], "力量コード": ["SKILL001", "SKILL002"]}
        )
        issues = monitor.check_timeliness(df)
        assert len(issues) > 0
        assert issues[0].severity == Severity.LOW
        assert "取得日カラムが存在しない" in issues[0].title

    def test_invalid_date_format(self, monitor):
        """無効な日付形式"""
        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M002", "M003"],
                "力量コード": ["SKILL001", "SKILL002", "SKILL003"],
                "取得日": ["2024/01/15", "invalid-date", "2024-99-99"],
            }
        )
        issues = monitor.check_timeliness(df)
        invalid_date_issues = [i for i in issues if "無効な取得日" in i.title]
        assert len(invalid_date_issues) > 0
        assert invalid_date_issues[0].severity == Severity.MEDIUM

    def test_stale_data(self, monitor):
        """古いデータが多い場合"""
        today = datetime.now()
        old_date = (today - timedelta(days=200)).strftime("%Y/%m/%d")

        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M002", "M003", "M004", "M005"] * 4,
                "力量コード": ["SKILL001"] * 20,
                "取得日": [old_date] * 20,  # 全て200日前
            }
        )

        issues = monitor.check_timeliness(df)
        stale_issues = [i for i in issues if "古いデータ" in i.title]
        assert len(stale_issues) > 0
        assert stale_issues[0].severity in [Severity.MEDIUM, Severity.HIGH]

    def test_future_dates(self, monitor):
        """未来の日付がある場合"""
        today = datetime.now()
        future_date = (today + timedelta(days=30)).strftime("%Y/%m/%d")

        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M002"],
                "力量コード": ["SKILL001", "SKILL002"],
                "取得日": ["2024/01/15", future_date],
            }
        )

        issues = monitor.check_timeliness(df)
        future_issues = [i for i in issues if "未来の取得日" in i.title]
        assert len(future_issues) > 0
        assert future_issues[0].severity == Severity.HIGH


class TestRapidAcquisitionCheck:
    """高速習得チェックのテスト"""

    def test_no_rapid_acquisition(self, monitor, sample_member_competence):
        """通常の学習ペース"""
        issues = monitor.check_rapid_acquisition(sample_member_competence)
        assert len(issues) == 0

    def test_rapid_acquisition_detected(self, monitor):
        """1週間に多数のスキル習得"""
        base_date = datetime(2024, 1, 15)
        dates = [(base_date + timedelta(days=i)).strftime("%Y/%m/%d") for i in range(5)]

        df = pd.DataFrame(
            {
                "メンバーコード": ["M001"] * 5,
                "力量コード": [f"SKILL{i:03d}" for i in range(1, 6)],
                "取得日": dates,  # 5日間で5スキル（週3スキル以上）
            }
        )

        issues = monitor.check_rapid_acquisition(df)
        rapid_issues = [i for i in issues if "異常に早いスキル習得" in i.title]
        assert len(rapid_issues) > 0
        assert rapid_issues[0].severity == Severity.MEDIUM

    def test_missing_date_column_no_issues(self, monitor):
        """取得日カラムがない場合は問題なし"""
        df = pd.DataFrame(
            {"メンバーコード": ["M001", "M002"], "力量コード": ["SKILL001", "SKILL002"]}
        )
        issues = monitor.check_rapid_acquisition(df)
        assert len(issues) == 0


class TestCheckAll:
    """統合チェックのテスト"""

    def test_check_all_clean_data(
        self, monitor, sample_member_competence, sample_competence_master
    ):
        """正常なデータの場合"""
        report = monitor.check_all(sample_member_competence, sample_competence_master)

        assert isinstance(report, DataQualityReport)
        assert report.total_records == len(sample_member_competence)
        assert report.total_issues >= 0  # 日付によっては古いデータ警告が出る可能性
        assert "data_shape" in report.summary
        assert "total_members" in report.summary
        assert "total_skills" in report.summary

    def test_check_all_with_multiple_issues(self, monitor):
        """複数の問題がある場合"""
        today = datetime.now()
        old_date = (today - timedelta(days=200)).strftime("%Y/%m/%d")

        df = pd.DataFrame(
            {
                "メンバーコード": ["M001", "M001", "M002", np.nan],
                "力量コード": ["SKILL001", "SKILL001", "SKILL002", "SKILL003"],
                "レベル": [10, 3, np.nan, 4],  # 範囲外とnull
                "取得日": [old_date, old_date, "invalid", "2024/01/15"],
            }
        )

        report = monitor.check_all(df)

        assert report.total_issues > 0
        assert len(report.issues) > 0

        # 重大度別の集計
        assert "issues_by_severity" in report.__dict__

        # CRITICALまたはHIGH問題がある
        high_priority = report.get_high_issues()
        assert len(high_priority) > 0

    def test_report_critical_issues_method(self, monitor):
        """CRITICALな問題のフィルタリング"""
        df = pd.DataFrame(
            {"メンバーコード": ["M001", "M002"], "力量コード": ["SKILL001", "SKILL002"]}
        )
        # 力量コードカラムを削除してCRITICALエラーを発生させる
        df_invalid = df.drop(columns=["力量コード"])

        report = monitor.check_all(df_invalid)
        critical_issues = report.get_critical_issues()

        assert len(critical_issues) > 0
        assert all(issue.severity == Severity.CRITICAL for issue in critical_issues)


class TestDataQualityIssue:
    """DataQualityIssueのテスト"""

    def test_issue_creation(self):
        """問題オブジェクトの作成"""
        issue = DataQualityIssue(
            category="completeness",
            severity=Severity.HIGH,
            title="テスト問題",
            message="これはテストです",
            affected_records=10,
            details={"key": "value"},
            recommendations=["修正してください"],
        )

        assert issue.category == "completeness"
        assert issue.severity == Severity.HIGH
        assert issue.title == "テスト問題"
        assert issue.affected_records == 10
        assert "key" in issue.details
        assert len(issue.recommendations) == 1


class TestDataQualityReport:
    """DataQualityReportのテスト"""

    def test_report_creation(self):
        """レポートオブジェクトの作成"""
        issues = [
            DataQualityIssue(
                category="completeness",
                severity=Severity.CRITICAL,
                title="Critical Issue",
                message="Test",
                affected_records=5,
            ),
            DataQualityIssue(
                category="consistency",
                severity=Severity.HIGH,
                title="High Issue",
                message="Test",
                affected_records=3,
            ),
            DataQualityIssue(
                category="anomaly",
                severity=Severity.MEDIUM,
                title="Medium Issue",
                message="Test",
                affected_records=2,
            ),
        ]

        report = DataQualityReport(
            timestamp=datetime.now(),
            total_records=100,
            total_issues=3,
            issues_by_severity={"CRITICAL": 1, "HIGH": 1, "MEDIUM": 1},
            issues=issues,
            summary={"test": "data"},
        )

        assert report.total_records == 100
        assert report.total_issues == 3
        assert len(report.issues) == 3

        # メソッドのテスト
        critical = report.get_critical_issues()
        assert len(critical) == 1

        high_priority = report.get_high_issues()
        assert len(high_priority) == 2  # CRITICAL + HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
