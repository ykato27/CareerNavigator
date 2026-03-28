"""
データ変換

スキルノートのデータを推薦システム用に変換
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from skillnote_recommendation.core.config import Config
from skillnote_recommendation.utils.data_normalizers import DataNormalizer
from skillnote_recommendation.utils.data_validators import DataValidator, ValidationError
from skillnote_recommendation.core.error_handlers import ErrorHandler


logger = logging.getLogger(__name__)


class DataTransformer:
    """
    データ変換クラス

    生データを推薦システムで使用可能な形式に変換します。
    データの正規化、検証、変換を担当します。
    """

    def __init__(self):
        """Initialize DataTransformer with normalizer and validator."""
        self.normalizer = DataNormalizer()
        self.validator = DataValidator()
        self.config = Config()

    @staticmethod
    def find_column(df: pd.DataFrame, keyword: str) -> str:
        """
        カラム名を動的に検出

        Args:
            df: DataFrame
            keyword: カラム名に含まれるキーワード（例：'メンバーコード', '職種'）

        Returns:
            見つかったカラム名（見つからない場合はNone）
        """
        for col in df.columns:
            if keyword in col:
                return col
        return None

    @staticmethod
    def normalize_level(level_value, competence_type: str) -> int:
        """
        レベル値を正規化

        Args:
            level_value: 元のレベル値
            competence_type: 力量タイプ（SKILL/EDUCATION/LICENSE）

        Returns:
            正規化されたレベル（0-5）
        """
        if competence_type == "SKILL":
            try:
                return int(level_value)
            except:
                return 0
        else:  # EDUCATION or LICENSE
            if pd.isna(level_value) or str(level_value).strip() == "":
                return 0
            elif str(level_value).strip() == "●":
                return 1
            else:
                return 0

    def create_competence_master(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        統合力量マスタを作成

        Args:
            data: 読み込んだデータの辞書

        Returns:
            統合力量マスタ
        """
        logger.info("\n" + "=" * 80)
        logger.info("統合力量マスタ作成")
        logger.info("=" * 80)

        # 各力量タイプのマスタを整形
        # カラム名を動的に検出
        def extract_competence_cols(df):
            """力量マスタのカラムを動的に抽出して標準名に正規化"""
            result_cols = []
            col_mapping = {}

            competence_code_col = self.find_column(df, "力量コード")
            competence_name_col = self.find_column(df, "力量名")
            category_code_col = self.find_column(df, "力量カテゴリーコード")
            summary_col = self.find_column(df, "概要")

            if competence_code_col:
                result_cols.append(competence_code_col)
                col_mapping[competence_code_col] = "力量コード"
            if competence_name_col:
                result_cols.append(competence_name_col)
                col_mapping[competence_name_col] = "力量名"
            if category_code_col:
                result_cols.append(category_code_col)
                col_mapping[category_code_col] = "力量カテゴリーコード"
            if summary_col:
                result_cols.append(summary_col)
                col_mapping[summary_col] = "概要"

            extracted = df[result_cols].copy()
            extracted = extracted.rename(columns=col_mapping)
            return extracted

        skills = extract_competence_cols(data["skills"])
        skills["力量タイプ"] = "SKILL"
        skills["レベル範囲"] = "1-5"

        education = extract_competence_cols(data["education"])
        education["力量タイプ"] = "EDUCATION"
        education["レベル範囲"] = "●"

        license_data = extract_competence_cols(data["license"])
        license_data["力量タイプ"] = "LICENSE"
        license_data["レベル範囲"] = "●"

        # 統合
        competence_master = pd.concat([skills, education, license_data], ignore_index=True)

        # 主カテゴリコードを抽出
        competence_master["力量カテゴリーコード_主"] = competence_master[
            "力量カテゴリーコード"
        ].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "")

        # カテゴリ名マッピングを作成
        category_names = self._create_category_names(data["categories"])
        competence_master["力量カテゴリー名"] = competence_master["力量カテゴリーコード_主"].map(
            category_names
        )

        logger.info("\n統合完了: %d件", len(competence_master))
        logger.info(
            "  - SKILL: %d件", len(competence_master[competence_master["力量タイプ"] == "SKILL"])
        )
        logger.info(
            "  - EDUCATION: %d件",
            len(competence_master[competence_master["力量タイプ"] == "EDUCATION"]),
        )
        logger.info(
            "  - LICENSE: %d件",
            len(competence_master[competence_master["力量タイプ"] == "LICENSE"]),
        )

        return competence_master

    def _create_category_names(self, df_categories: pd.DataFrame) -> dict:
        """
        カテゴリコードからカテゴリ名へのマッピングを作成

        Args:
            df_categories: カテゴリマスタ

        Returns:
            カテゴリ名マッピング辞書
        """
        category_names = {}

        for _, row in df_categories.iterrows():
            code = row.iloc[0]
            names = []
            for i in range(1, len(row)):
                val = row.iloc[i]
                if not pd.isna(val) and str(val).strip() != "":
                    names.append(str(val))
            category_names[code] = " > ".join(names) if names else ""

        return category_names

    def create_member_competence(
        self, data: Dict[str, pd.DataFrame], competence_master: pd.DataFrame
    ) -> Tuple[pd.DataFrame, list]:
        """
        メンバー習得力量データを作成

        Args:
            data: 読み込んだデータの辞書
            competence_master: 統合力量マスタ

        Returns:
            (メンバー習得力量データ, 有効なメンバーコードリスト)
        """
        logger.info("\n" + "=" * 80)
        logger.info("メンバー習得力量データ作成")
        logger.info("=" * 80)

        # 習得力量データの確認
        acquired_df = data["acquired"].copy()
        logger.info("\n保有力量データ:")
        logger.info("  総行数: %d", len(acquired_df))
        logger.info("  カラム: %s", list(acquired_df.columns))

        # カラム名を動的に検出
        member_code_col_acquired = self.find_column(acquired_df, "メンバーコード")
        competence_code_col = self.find_column(acquired_df, "力量コード")
        competence_type_col = self.find_column(acquired_df, "力量タイプ")
        level_col = self.find_column(acquired_df, "レベル")

        # 必須カラムの存在確認
        missing_cols = []
        if not member_code_col_acquired:
            missing_cols.append("メンバーコード")
        if not competence_code_col:
            missing_cols.append("力量コード")
        if not competence_type_col:
            missing_cols.append("力量タイプ")
        if not level_col:
            missing_cols.append("レベル")

        if missing_cols:
            raise ValueError(f"保有力量データに必須カラムがありません: {missing_cols}")

        logger.info("\n検出されたカラム名:")
        logger.info("  メンバーコード: %s", member_code_col_acquired)
        logger.info("  力量コード: %s", competence_code_col)
        logger.info("  力量タイプ: %s", competence_type_col)
        logger.info("  レベル: %s", level_col)

        # メンバーコードを正規化（新しいnormalizerを使用）
        logger.info("\nメンバーコードを正規化中...")
        acquired_df[member_code_col_acquired] = acquired_df[member_code_col_acquired].apply(
            self.normalizer.normalize_member_code
        )
        logger.info("  保有力量データのメンバーコードを正規化しました")

        # 有効なメンバーを抽出
        members_df = data["members"].copy()
        member_code_col_members = self.find_column(members_df, "メンバーコード")
        member_name_col = self.find_column(members_df, "メンバー名")

        if not member_code_col_members or not member_name_col:
            raise ValueError("メンバーマスタに必須カラムがありません")

        members_df[member_code_col_members] = members_df[member_code_col_members].apply(
            self.normalizer.normalize_member_code
        )

        # 無効な名前パターンを設定から取得
        invalid_patterns = self.config.VALIDATION_PARAMS["invalid_name_patterns"]
        pattern = "|".join(invalid_patterns)

        valid_members = members_df[
            (~members_df[member_name_col].str.contains(pattern, case=False, na=False))
            & (members_df[member_code_col_members] != "")
        ][member_code_col_members].unique()

        logger.info("\n有効なメンバー数: %d名", len(valid_members))
        if len(valid_members) > 0:
            logger.info("  有効なメンバーの例（最初の5名）: %s", list(valid_members[:5]))

        # 保有力量データに含まれるメンバーコードを確認
        acquired_member_codes = acquired_df[member_code_col_acquired].unique()
        logger.info("\n保有力量データに含まれるメンバー数: %d名", len(acquired_member_codes))
        if len(acquired_member_codes) > 0:
            logger.info(
                "  保有力量データのメンバーの例（最初の5名）: %s", list(acquired_member_codes[:5])
            )

        # 一致するメンバーを確認
        matching_members = set(valid_members) & set(acquired_member_codes)
        logger.info("\n一致するメンバー数: %d名", len(matching_members))
        if len(matching_members) == 0:
            logger.warning("  ⚠ 有効なメンバーと保有力量データのメンバーが一致しません！")
            logger.warning(
                "  メンバーマスタと保有力量データのメンバーコードの形式を確認してください"
            )

        # 習得力量データ
        member_competence = acquired_df[
            acquired_df[member_code_col_acquired].isin(valid_members)
        ].copy()

        logger.info("\n有効メンバーでフィルタ後: %d件", len(member_competence))

        if member_competence.empty:
            logger.warning("  ⚠ 有効なメンバーの習得力量データがありません")
            logger.warning("  保有力量データの先頭5件:")
            logger.warning("\n%s", acquired_df.head().to_string())

            # 空のDataFrameでも必要なカラムを持つ構造を返す
            # 動的に検出されたカラム名を使用
            empty_df = pd.DataFrame(
                columns=[
                    member_code_col_acquired,
                    competence_code_col,
                    competence_type_col,
                    level_col,
                    "正規化レベル",
                    "力量名",
                    "力量カテゴリー名",
                ]
            )
            return empty_df, valid_members.tolist()

        # レベル正規化
        member_competence["正規化レベル"] = member_competence.apply(
            lambda row: self.normalize_level(row[level_col], row[competence_type_col]), axis=1
        )

        # 力量マスタと結合
        # 力量マスタのカラム名も動的に検出
        competence_code_col_master = self.find_column(competence_master, "力量コード")
        competence_name_col_master = self.find_column(competence_master, "力量名")
        competence_type_col_master = self.find_column(competence_master, "力量タイプ")
        competence_category_col_master = self.find_column(competence_master, "力量カテゴリー名")

        merge_cols = [competence_code_col_master]
        if competence_name_col_master:
            merge_cols.append(competence_name_col_master)
        if competence_type_col_master:
            merge_cols.append(competence_type_col_master)
        if competence_category_col_master:
            merge_cols.append(competence_category_col_master)

        member_competence = member_competence.merge(
            competence_master[merge_cols],
            left_on=competence_code_col,
            right_on=competence_code_col_master,
            how="left",
            suffixes=("", "_master"),
        )

        logger.info("力量マスタと結合後: %d件", len(member_competence))

        # カラム名を統一（標準名に正規化）
        column_mapping = {
            member_code_col_acquired: "メンバーコード",
            competence_code_col: "力量コード",
            competence_type_col: "力量タイプ",
            level_col: "レベル",
        }
        member_competence = member_competence.rename(columns=column_mapping)

        return member_competence, valid_members.tolist()

    def create_skill_matrix(self, member_competence: pd.DataFrame) -> pd.DataFrame:
        """
        メンバー×力量マトリックスを作成

        Args:
            member_competence: メンバー習得力量データ

        Returns:
            メンバー×力量マトリックス
        """
        logger.info("\n" + "=" * 80)
        logger.info("メンバー×力量マトリックス作成")
        logger.info("=" * 80)

        # 空データの場合は空のマトリックスを返す
        if member_competence.empty:
            logger.warning("  ⚠ メンバー習得力量データが空のため、空のマトリックスを返します")
            return pd.DataFrame()

        # 必須カラムの確認
        required_columns = ["メンバーコード", "力量コード", "正規化レベル"]
        missing_columns = [col for col in required_columns if col not in member_competence.columns]
        if missing_columns:
            logger.error("  ⚠ 必須カラムが不足しています: %s", missing_columns)
            logger.error("  利用可能なカラム: %s", list(member_competence.columns))
            raise ValueError(
                f"メンバー習得力量データに必須カラムが不足しています: {missing_columns}"
            )

        skill_matrix = member_competence.pivot_table(
            index="メンバーコード", columns="力量コード", values="正規化レベル", fill_value=0
        )

        logger.info(
            "\nマトリックスサイズ: %d名 × %d力量",
            skill_matrix.shape[0],
            skill_matrix.shape[1],
        )

        return skill_matrix

    def clean_members_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        メンバーマスタをクリーンにする

        Args:
            data: 読み込んだデータの辞書

        Returns:
            クリーンなメンバーマスタ
        """
        members_df = data["members"].copy()

        # カラム名を動的に検出
        member_code_col = self.find_column(members_df, "メンバーコード")
        member_name_col = self.find_column(members_df, "メンバー名")

        if not member_code_col or not member_name_col:
            raise ValueError("メンバーマスタに必須カラムがありません")

        # メンバーコードを正規化（新しいnormalizerを使用）
        members_df[member_code_col] = members_df[member_code_col].apply(
            self.normalizer.normalize_member_code
        )

        # 無効な名前パターンを設定から取得
        invalid_patterns = self.config.VALIDATION_PARAMS["invalid_name_patterns"]
        pattern = "|".join(invalid_patterns)

        members_clean = members_df[
            (~members_df[member_name_col].str.contains(pattern, case=False, na=False))
            & (members_df[member_code_col] != "")
        ].copy()

        # 必要なカラムキーワードを定義
        column_keywords = [
            "メンバーコード",
            "メンバー名",
            "よみがな",
            "生年月日",
            "性別",
            "入社年月日",
            "社員区分",
            "役職",
            "職能・等級",
            "職種",
        ]

        # 存在するカラムを動的に検出して選択
        available_columns = []
        for keyword in column_keywords:
            col = self.find_column(members_clean, keyword)
            if col:
                available_columns.append(col)

        selected_df = members_clean[available_columns].copy()

        # カラム名を標準名に正規化
        column_mapping = {}
        for col in selected_df.columns:
            for keyword in column_keywords:
                if keyword in col:
                    column_mapping[col] = keyword
                    break

        selected_df = selected_df.rename(columns=column_mapping)

        return selected_df
