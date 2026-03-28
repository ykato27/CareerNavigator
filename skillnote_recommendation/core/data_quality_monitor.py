"""
データ品質モニタリングモジュール

データの完全性、一貫性、適時性、異常値を検出し、
データ品質の問題を早期に発見します。

Phase 1: Completeness, Consistency, Anomaly Detection (Duplicates)
Phase 2: Timeliness, Anomaly Detection (Rapid Acquisition)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Severity(Enum):
    """問題の重大度"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class DataQualityIssue:
    """データ品質の問題"""

    category: str  # 'completeness', 'consistency', 'timeliness', 'anomaly'
    severity: Severity
    title: str
    message: str
    affected_records: int
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """データ品質レポート"""

    timestamp: datetime
    total_records: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues: List[DataQualityIssue]
    summary: Dict[str, Any]

    def get_critical_issues(self) -> List[DataQualityIssue]:
        """重大な問題のみを取得"""
        return [issue for issue in self.issues if issue.severity == Severity.CRITICAL]

    def get_high_issues(self) -> List[DataQualityIssue]:
        """高優先度の問題を取得"""
        return [
            issue for issue in self.issues if issue.severity in [Severity.CRITICAL, Severity.HIGH]
        ]


class DataQualityMonitor:
    """
    データ品質モニタリングクラス

    データの品質をチェックし、問題を検出してレポートを生成します。
    """

    def __init__(
        self,
        missing_threshold: float = 0.05,  # 欠損率の閾値（5%）
        staleness_days: int = 180,  # データ鮮度の閾値（6ヶ月）
        max_skills_per_week: int = 3,  # 週あたりの最大スキル習得数
        skill_dependencies: Optional[Dict[str, List[str]]] = None,
    ):
        """
        初期化

        Args:
            missing_threshold: 欠損率の閾値（この値を超えると警告）
            staleness_days: データ鮮度の閾値（この日数以上古いと警告）
            max_skills_per_week: 週あたりの最大スキル習得数
            skill_dependencies: スキルの依存関係 {上級スキル: [前提スキル]}
        """
        self.missing_threshold = missing_threshold
        self.staleness_days = staleness_days
        self.max_skills_per_week = max_skills_per_week
        self.skill_dependencies = skill_dependencies or {}

    def check_all(
        self,
        member_competence: pd.DataFrame,
        competence_master: Optional[pd.DataFrame] = None,
        members: Optional[pd.DataFrame] = None,
    ) -> DataQualityReport:
        """
        全てのデータ品質チェックを実行

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタ（オプション）
            members: メンバーマスタ（オプション）

        Returns:
            データ品質レポート
        """
        logger.info("=" * 80)
        logger.info("データ品質チェックを開始")
        logger.info("=" * 80)

        issues = []

        # Phase 1: Completeness（完全性）
        logger.info("\n[Phase 1] Completeness チェック...")
        completeness_issues = self.check_completeness(member_competence)
        issues.extend(completeness_issues)
        logger.info(f"  → {len(completeness_issues)}件の問題を検出")

        # Phase 1: Consistency（一貫性）
        logger.info("\n[Phase 1] Consistency チェック...")
        consistency_issues = self.check_consistency(member_competence, competence_master)
        issues.extend(consistency_issues)
        logger.info(f"  → {len(consistency_issues)}件の問題を検出")

        # Phase 1: Anomaly（異常値 - 重複）
        logger.info("\n[Phase 1] Anomaly（重複）チェック...")
        duplicate_issues = self.check_duplicates(member_competence)
        issues.extend(duplicate_issues)
        logger.info(f"  → {len(duplicate_issues)}件の問題を検出")

        # Phase 2: Timeliness（適時性）
        logger.info("\n[Phase 2] Timeliness チェック...")
        timeliness_issues = self.check_timeliness(member_competence)
        issues.extend(timeliness_issues)
        logger.info(f"  → {len(timeliness_issues)}件の問題を検出")

        # Phase 2: Anomaly（異常値 - 高速習得）
        logger.info("\n[Phase 2] Anomaly（高速習得）チェック...")
        rapid_acquisition_issues = self.check_rapid_acquisition(member_competence)
        issues.extend(rapid_acquisition_issues)
        logger.info(f"  → {len(rapid_acquisition_issues)}件の問題を検出")

        # レポートを生成
        report = self._generate_report(member_competence, issues)

        logger.info("\n" + "=" * 80)
        logger.info(f"データ品質チェック完了: {len(issues)}件の問題を検出")
        logger.info(f"  CRITICAL: {report.issues_by_severity.get('CRITICAL', 0)}件")
        logger.info(f"  HIGH: {report.issues_by_severity.get('HIGH', 0)}件")
        logger.info(f"  MEDIUM: {report.issues_by_severity.get('MEDIUM', 0)}件")
        logger.info(f"  LOW: {report.issues_by_severity.get('LOW', 0)}件")
        logger.info("=" * 80)

        return report

    def check_completeness(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """
        完全性チェック: 欠損値の検出

        Args:
            df: メンバー習得力量データ

        Returns:
            検出された問題のリスト
        """
        issues = []

        # 必須カラムのチェック
        required_columns = ["メンバーコード", "力量コード"]
        for col in required_columns:
            if col not in df.columns:
                issues.append(
                    DataQualityIssue(
                        category="completeness",
                        severity=Severity.CRITICAL,
                        title=f"必須カラム '{col}' が存在しません",
                        message=f"データに必須カラム '{col}' がありません。データ構造を確認してください。",
                        affected_records=len(df),
                        recommendations=[
                            f"CSVファイルに '{col}' カラムを追加してください",
                            "データのスキーマを確認してください",
                        ],
                    )
                )
                return issues  # これ以上チェックできない

        # 欠損値の分析
        total_rows = len(df)
        missing_analysis = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_ratio = missing_count / total_rows
                missing_analysis[col] = {"count": int(missing_count), "ratio": float(missing_ratio)}

                # 閾値を超えた欠損
                if missing_ratio > self.missing_threshold:
                    severity = Severity.HIGH if missing_ratio > 0.2 else Severity.MEDIUM

                    issues.append(
                        DataQualityIssue(
                            category="completeness",
                            severity=severity,
                            title=f"カラム '{col}' の欠損率が高い",
                            message=f"カラム '{col}' に {missing_count}件（{missing_ratio:.1%}）の欠損があります。"
                            f"閾値: {self.missing_threshold:.1%}",
                            affected_records=int(missing_count),
                            details={
                                "column": col,
                                "missing_count": int(missing_count),
                                "missing_ratio": float(missing_ratio),
                                "threshold": float(self.missing_threshold),
                            },
                            recommendations=[
                                f"'{col}' の欠損値を補完するか、レコードを削除してください",
                                "データ入力プロセスを見直してください",
                            ],
                        )
                    )

        # 完全に空のレコード
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            issues.append(
                DataQualityIssue(
                    category="completeness",
                    severity=Severity.HIGH,
                    title="完全に空のレコードが存在",
                    message=f"{empty_rows}件の完全に空のレコードがあります。",
                    affected_records=int(empty_rows),
                    recommendations=["空のレコードを削除してください"],
                )
            )

        return issues

    def check_consistency(
        self, member_competence: pd.DataFrame, competence_master: Optional[pd.DataFrame] = None
    ) -> List[DataQualityIssue]:
        """
        一貫性チェック: 論理的整合性の検証

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタ

        Returns:
            検出された問題のリスト
        """
        issues = []

        # レベル値の範囲チェック
        if "レベル" in member_competence.columns or "正規化レベル" in member_competence.columns:
            level_col = "正規化レベル" if "正規化レベル" in member_competence.columns else "レベル"

            # 数値型に変換（エラーは無視）
            levels = pd.to_numeric(member_competence[level_col], errors="coerce")

            # 範囲外のレベル
            invalid_levels = member_competence[((levels < 0) | (levels > 5)) & levels.notna()]

            if len(invalid_levels) > 0:
                issues.append(
                    DataQualityIssue(
                        category="consistency",
                        severity=Severity.HIGH,
                        title="レベル値が範囲外",
                        message=f"{len(invalid_levels)}件のレコードでレベルが範囲外（0-5）です。",
                        affected_records=len(invalid_levels),
                        details={
                            "invalid_values": invalid_levels[level_col].value_counts().to_dict(),
                            "expected_range": "0-5",
                        },
                        recommendations=[
                            "レベル値を0-5の範囲に修正してください",
                            "データ入力時のバリデーションを追加してください",
                        ],
                    )
                )

            # レベル0だが取得日がある
            if "取得日" in member_competence.columns:
                zero_with_date = member_competence[
                    (levels == 0) & (member_competence["取得日"].notna())
                ]

                if len(zero_with_date) > 0:
                    issues.append(
                        DataQualityIssue(
                            category="consistency",
                            severity=Severity.LOW,
                            title="レベル0だが取得日が記録されている",
                            message=f"{len(zero_with_date)}件のレコードでレベル0だが取得日があります。",
                            affected_records=len(zero_with_date),
                            recommendations=[
                                "レベル0のレコードから取得日を削除するか、レベルを修正してください"
                            ],
                        )
                    )

        # 前提スキルの依存関係チェック
        if self.skill_dependencies and "取得日" in member_competence.columns:
            violations = self._check_prerequisite_violations(member_competence)

            if violations:
                # 重大度別に集計
                high_severity_count = sum(1 for v in violations if v["severity"] == "HIGH")

                if high_severity_count > 0:
                    issues.append(
                        DataQualityIssue(
                            category="consistency",
                            severity=Severity.HIGH,
                            title="前提スキルの依存関係違反",
                            message=f"{high_severity_count}件で前提スキルが未習得のまま上級スキルを習得しています。",
                            affected_records=high_severity_count,
                            details={"violations": violations[:5]},  # 最初の5件のみ
                            recommendations=[
                                "前提スキルの取得状況を確認してください",
                                "データ入力の順序を見直してください",
                            ],
                        )
                    )

        # 時系列の整合性チェック
        if "取得日" in member_competence.columns:
            temporal_issues = self._check_temporal_consistency(member_competence)
            issues.extend(temporal_issues)

        return issues

    def check_duplicates(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """
        重複レコードの検出

        Args:
            df: メンバー習得力量データ

        Returns:
            検出された問題のリスト
        """
        issues = []

        # 完全重複のチェック
        key_columns = ["メンバーコード", "力量コード"]
        if "取得日" in df.columns:
            key_columns.append("取得日")

        # キーカラムが全て存在するか確認
        if all(col in df.columns for col in key_columns):
            duplicates = df[df.duplicated(subset=key_columns, keep=False)]

            if len(duplicates) > 0:
                duplicate_groups = duplicates.groupby(key_columns).size()
                duplicate_count = len(duplicate_groups)

                issues.append(
                    DataQualityIssue(
                        category="anomaly",
                        severity=Severity.MEDIUM,
                        title="重複レコードを検出",
                        message=f"{len(duplicates)}件の重複レコード（{duplicate_count}グループ）を検出しました。",
                        affected_records=len(duplicates),
                        details={
                            "duplicate_groups": int(duplicate_count),
                            "top_duplicates": duplicate_groups.head(10).to_dict(),
                        },
                        recommendations=[
                            "重複レコードを削除してください",
                            "データ入力時の重複チェックを追加してください",
                        ],
                    )
                )

        # 同じメンバー×力量の複数レコード（取得日が異なる）
        if "取得日" in df.columns:
            multi_acquisition = df.groupby(["メンバーコード", "力量コード"]).size()
            multi_acquisition = multi_acquisition[multi_acquisition > 1]

            if len(multi_acquisition) > 0:
                issues.append(
                    DataQualityIssue(
                        category="anomaly",
                        severity=Severity.LOW,
                        title="同じスキルを複数回習得",
                        message=f"{len(multi_acquisition)}件で同じスキルが複数の取得日で記録されています。",
                        affected_records=multi_acquisition.sum(),
                        details={"examples": multi_acquisition.head(5).to_dict()},
                        recommendations=[
                            "最新の取得日のみを残すか、レベル更新として扱ってください"
                        ],
                    )
                )

        return issues

    def check_timeliness(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """
        適時性チェック: データの鮮度を確認

        Args:
            df: メンバー習得力量データ

        Returns:
            検出された問題のリスト
        """
        issues = []

        if "取得日" not in df.columns:
            issues.append(
                DataQualityIssue(
                    category="timeliness",
                    severity=Severity.LOW,
                    title="取得日カラムが存在しない",
                    message="取得日カラムがないため、データ鮮度をチェックできません。",
                    affected_records=len(df),
                    recommendations=["取得日カラムを追加してください"],
                )
            )
            return issues

        # 取得日をdatetime型に変換
        dates = pd.to_datetime(df["取得日"], errors="coerce")

        # 無効な日付
        invalid_dates_count = dates.isnull().sum()
        if invalid_dates_count > 0:
            issues.append(
                DataQualityIssue(
                    category="timeliness",
                    severity=Severity.MEDIUM,
                    title="無効な取得日",
                    message=f"{invalid_dates_count}件で取得日が無効または解析不能です。",
                    affected_records=int(invalid_dates_count),
                    recommendations=[
                        "取得日の形式を確認してください（YYYY/MM/DD または YYYY-MM-DD）"
                    ],
                )
            )

        # データの鮮度チェック
        today = datetime.now()
        threshold_date = today - timedelta(days=self.staleness_days)

        valid_dates = dates[dates.notna()]
        if len(valid_dates) > 0:
            stale_records = valid_dates[valid_dates < threshold_date]

            if len(stale_records) > 0:
                staleness_ratio = len(stale_records) / len(valid_dates)
                oldest_date = valid_dates.min()

                severity = Severity.HIGH if staleness_ratio > 0.5 else Severity.MEDIUM

                issues.append(
                    DataQualityIssue(
                        category="timeliness",
                        severity=severity,
                        title="古いデータが多い",
                        message=f"{len(stale_records)}件（{staleness_ratio:.1%}）のレコードが"
                        f"{self.staleness_days}日以上古いデータです。",
                        affected_records=len(stale_records),
                        details={
                            "staleness_ratio": float(staleness_ratio),
                            "threshold_days": self.staleness_days,
                            "oldest_date": str(oldest_date.date()),
                            "oldest_days_ago": (today - oldest_date).days,
                        },
                        recommendations=[
                            "データを最新の状態に更新してください",
                            "定期的なデータ更新プロセスを確立してください",
                        ],
                    )
                )

            # 未来の日付
            future_dates = valid_dates[valid_dates > today]
            if len(future_dates) > 0:
                issues.append(
                    DataQualityIssue(
                        category="timeliness",
                        severity=Severity.HIGH,
                        title="未来の取得日",
                        message=f"{len(future_dates)}件で取得日が未来日付です。",
                        affected_records=len(future_dates),
                        details={"latest_future_date": str(future_dates.max().date())},
                        recommendations=["取得日を現在または過去の日付に修正してください"],
                    )
                )

        return issues

    def check_rapid_acquisition(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """
        高速習得の検出: 短期間での大量スキル習得をチェック

        Args:
            df: メンバー習得力量データ

        Returns:
            検出された問題のリスト
        """
        issues = []

        if "取得日" not in df.columns:
            return issues

        # 取得日をdatetime型に変換
        temp_df = df.copy()
        temp_df["取得日_dt"] = pd.to_datetime(temp_df["取得日"], errors="coerce")
        temp_df = temp_df[temp_df["取得日_dt"].notna()]

        if len(temp_df) == 0:
            return issues

        # メンバーごとにチェック
        rapid_learners = []

        for member in temp_df["メンバーコード"].unique():
            member_data = temp_df[temp_df["メンバーコード"] == member].sort_values("取得日_dt")

            if len(member_data) < 2:
                continue

            # 7日間の移動窓でスキル数をカウント
            for i, row in member_data.iterrows():
                current_date = row["取得日_dt"]
                window_start = current_date - timedelta(days=7)

                skills_in_window = member_data[
                    (member_data["取得日_dt"] >= window_start)
                    & (member_data["取得日_dt"] <= current_date)
                ]

                if len(skills_in_window) > self.max_skills_per_week:
                    rapid_learners.append(
                        {
                            "member_code": member,
                            "period": f"{window_start.date()} ~ {current_date.date()}",
                            "skills_count": len(skills_in_window),
                            "threshold": self.max_skills_per_week,
                            "skills": skills_in_window["力量コード"].tolist(),
                        }
                    )
                    break  # メンバーごとに1回だけ報告

        if rapid_learners:
            issues.append(
                DataQualityIssue(
                    category="anomaly",
                    severity=Severity.MEDIUM,
                    title="異常に早いスキル習得を検出",
                    message=f"{len(rapid_learners)}人のメンバーが1週間に{self.max_skills_per_week}個以上の"
                    f"スキルを習得しています。データ入力の一括登録の可能性があります。",
                    affected_records=sum(r["skills_count"] for r in rapid_learners),
                    details={
                        "rapid_learners": rapid_learners[:5],  # 最初の5人のみ
                        "threshold_per_week": self.max_skills_per_week,
                    },
                    recommendations=[
                        "実際の取得日を確認してください",
                        "一括入力の場合は、より正確な取得日を記録してください",
                        "データ入力時に警告を表示するようにしてください",
                    ],
                )
            )

        return issues

    def _check_prerequisite_violations(self, member_competence: pd.DataFrame) -> List[Dict]:
        """前提スキルの依存関係違反をチェック"""
        violations = []

        for member in member_competence["メンバーコード"].unique():
            member_data = member_competence[member_competence["メンバーコード"] == member]
            acquired_skills = set(member_data["力量コード"])

            for advanced_skill, prerequisites in self.skill_dependencies.items():
                if advanced_skill in acquired_skills:
                    missing_prereqs = [
                        prereq for prereq in prerequisites if prereq not in acquired_skills
                    ]

                    if missing_prereqs:
                        violations.append(
                            {
                                "member_code": member,
                                "skill": advanced_skill,
                                "missing_prerequisites": missing_prereqs,
                                "severity": "HIGH",
                            }
                        )

        return violations

    def _check_temporal_consistency(
        self, member_competence: pd.DataFrame
    ) -> List[DataQualityIssue]:
        """時系列の整合性をチェック"""
        issues = []

        if not self.skill_dependencies:
            return issues

        df = member_competence.copy()
        df["取得日_dt"] = pd.to_datetime(df["取得日"], errors="coerce")
        df = df[df["取得日_dt"].notna()]

        temporal_violations = []

        for member in df["メンバーコード"].unique():
            member_data = df[df["メンバーコード"] == member]

            for skill, prerequisites in self.skill_dependencies.items():
                skill_rows = member_data[member_data["力量コード"] == skill]

                if skill_rows.empty:
                    continue

                skill_date = skill_rows.iloc[0]["取得日_dt"]

                for prereq in prerequisites:
                    prereq_rows = member_data[member_data["力量コード"] == prereq]

                    if not prereq_rows.empty:
                        prereq_date = prereq_rows.iloc[0]["取得日_dt"]

                        # 前提スキルより前に上級スキルを習得
                        if skill_date < prereq_date:
                            temporal_violations.append(
                                {
                                    "member": member,
                                    "skill": skill,
                                    "skill_date": str(skill_date.date()),
                                    "prerequisite": prereq,
                                    "prerequisite_date": str(prereq_date.date()),
                                }
                            )

        if temporal_violations:
            issues.append(
                DataQualityIssue(
                    category="consistency",
                    severity=Severity.MEDIUM,
                    title="時系列の依存関係違反",
                    message=f"{len(temporal_violations)}件で前提スキルより前に上級スキルを習得しています。",
                    affected_records=len(temporal_violations),
                    details={"violations": temporal_violations[:5]},
                    recommendations=[
                        "取得日の正確性を確認してください",
                        "データ入力の順序を見直してください",
                    ],
                )
            )

        return issues

    def _generate_report(
        self, df: pd.DataFrame, issues: List[DataQualityIssue]
    ) -> DataQualityReport:
        """レポートを生成"""
        issues_by_severity = {}
        for severity in Severity:
            count = sum(1 for issue in issues if issue.severity == severity)
            if count > 0:
                issues_by_severity[severity.value] = count

        summary = {
            "data_shape": df.shape,
            "total_members": (
                df["メンバーコード"].nunique() if "メンバーコード" in df.columns else 0
            ),
            "total_skills": df["力量コード"].nunique() if "力量コード" in df.columns else 0,
            "has_acquisition_date": "取得日" in df.columns,
        }

        return DataQualityReport(
            timestamp=datetime.now(),
            total_records=len(df),
            total_issues=len(issues),
            issues_by_severity=issues_by_severity,
            issues=issues,
            summary=summary,
        )
