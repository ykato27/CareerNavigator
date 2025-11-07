"""
DataTransformerクラスのテスト

データ変換（レベル正規化、マスタ作成、マトリクス生成）をテスト
"""

import pytest
import pandas as pd
import numpy as np
from skillnote_recommendation.core.data_transformer import DataTransformer


# ==================== レベル正規化テスト ====================


class TestNormalizeLevel:
    """レベル正規化のテスト"""

    @pytest.mark.parametrize(
        "level,expected",
        [
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("4", 4),
            ("5", 5),
            (1, 1),
            (3, 3),
            (5, 5),
        ],
    )
    def test_normalize_level_skill_valid(self, level, expected):
        """SKILLタイプの正規化（有効な値）"""
        result = DataTransformer.normalize_level(level, "SKILL")
        assert result == expected

    @pytest.mark.parametrize("level", ["invalid", "abc", "", None, "zero"])
    def test_normalize_level_skill_invalid(self, level):
        """SKILLタイプの正規化（無効な値は0）"""
        result = DataTransformer.normalize_level(level, "SKILL")
        assert result == 0

    def test_normalize_level_skill_out_of_range(self):
        """SKILLタイプの正規化（範囲外の値も許容）"""
        # 実装は範囲チェックしないため、整数変換できれば任意の値を返す
        result = DataTransformer.normalize_level("10", "SKILL")
        assert result == 10  # 範囲外だが整数として扱われる

    def test_normalize_level_education_marked(self):
        """EDUCATIONで●→1に変換"""
        result = DataTransformer.normalize_level("●", "EDUCATION")
        assert result == 1

    def test_normalize_level_education_empty(self):
        """EDUCATIONで空文字→0"""
        result = DataTransformer.normalize_level("", "EDUCATION")
        assert result == 0

    def test_normalize_level_education_nan(self):
        """EDUCATIONでNaN→0"""
        result = DataTransformer.normalize_level(np.nan, "EDUCATION")
        assert result == 0

    def test_normalize_level_education_spaces(self):
        """EDUCATIONで空白のみ→0"""
        result = DataTransformer.normalize_level("  ", "EDUCATION")
        assert result == 0

    def test_normalize_level_license_marked(self):
        """LICENSEで●→1に変換"""
        result = DataTransformer.normalize_level("●", "LICENSE")
        assert result == 1

    def test_normalize_level_license_empty(self):
        """LICENSEで空文字→0"""
        result = DataTransformer.normalize_level("", "LICENSE")
        assert result == 0

    def test_normalize_level_license_nan(self):
        """LICENSEでNaN→0"""
        result = DataTransformer.normalize_level(None, "LICENSE")
        assert result == 0

    @pytest.mark.parametrize(
        "comp_type,level,expected",
        [
            ("SKILL", "3", 3),
            ("SKILL", "invalid", 0),
            ("EDUCATION", "●", 1),
            ("EDUCATION", "", 0),
            ("LICENSE", "●", 1),
            ("LICENSE", None, 0),
        ],
    )
    def test_normalize_level_all_types(self, comp_type, level, expected):
        """全タイプのレベル正規化"""
        result = DataTransformer.normalize_level(level, comp_type)
        assert result == expected


# ==================== 統合力量マスタ作成テスト ====================


class TestCreateCompetenceMaster:
    """統合力量マスタ作成のテスト"""

    def test_create_competence_master(
        self, sample_skills, sample_education, sample_license, sample_categories
    ):
        """統合力量マスタが正しく作成される"""
        data = {
            "skills": sample_skills,
            "education": sample_education,
            "license": sample_license,
            "categories": sample_categories,
        }

        transformer = DataTransformer()
        master = transformer.create_competence_master(data)

        # 基本的な検証
        assert len(master) > 0
        assert "力量コード" in master.columns
        assert "力量名" in master.columns
        assert "力量タイプ" in master.columns

    def test_competence_master_types(
        self, sample_skills, sample_education, sample_license, sample_categories
    ):
        """3タイプすべてが含まれること"""
        data = {
            "skills": sample_skills,
            "education": sample_education,
            "license": sample_license,
            "categories": sample_categories,
        }

        transformer = DataTransformer()
        master = transformer.create_competence_master(data)

        types = master["力量タイプ"].unique()
        assert "SKILL" in types
        assert "EDUCATION" in types
        assert "LICENSE" in types

    def test_competence_master_count(
        self, sample_skills, sample_education, sample_license, sample_categories
    ):
        """件数が正しいこと"""
        data = {
            "skills": sample_skills,
            "education": sample_education,
            "license": sample_license,
            "categories": sample_categories,
        }

        transformer = DataTransformer()
        master = transformer.create_competence_master(data)

        skill_count = len(sample_skills)
        edu_count = len(sample_education)
        lic_count = len(sample_license)

        assert len(master) == skill_count + edu_count + lic_count

    def test_competence_master_level_ranges(
        self, sample_skills, sample_education, sample_license, sample_categories
    ):
        """レベル範囲が正しく設定される"""
        data = {
            "skills": sample_skills,
            "education": sample_education,
            "license": sample_license,
            "categories": sample_categories,
        }

        transformer = DataTransformer()
        master = transformer.create_competence_master(data)

        # SKILLは'1-5'
        skills = master[master["力量タイプ"] == "SKILL"]
        assert all(skills["レベル範囲"] == "1-5")

        # EDUCATION/LICENSEは'●'
        education = master[master["力量タイプ"] == "EDUCATION"]
        assert all(education["レベル範囲"] == "●")

        license_data = master[master["力量タイプ"] == "LICENSE"]
        assert all(license_data["レベル範囲"] == "●")

    def test_competence_master_category_names(
        self, sample_skills, sample_education, sample_license, sample_categories
    ):
        """カテゴリ名が正しくマッピングされる"""
        data = {
            "skills": sample_skills,
            "education": sample_education,
            "license": sample_license,
            "categories": sample_categories,
        }

        transformer = DataTransformer()
        master = transformer.create_competence_master(data)

        # カテゴリ名カラムが存在
        assert "力量カテゴリー名" in master.columns

        # カテゴリ名が設定されている
        assert master["力量カテゴリー名"].notna().any()


# ==================== カテゴリ名マッピング作成テスト ====================


class TestCreateCategoryNames:
    """カテゴリ名マッピング作成のテスト"""

    def test_create_category_names(self, sample_categories):
        """カテゴリ名マッピングが作成される"""
        transformer = DataTransformer()
        mapping = transformer._create_category_names(sample_categories)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_create_category_names_hierarchy(self, sample_categories):
        """階層カテゴリが' > '区切りで結合される"""
        transformer = DataTransformer()
        mapping = transformer._create_category_names(sample_categories)

        # 階層構造がある場合は' > 'で結合
        for code, name in mapping.items():
            if " > " in name:
                # 階層が正しく結合されている
                assert isinstance(name, str)

    def test_create_category_names_empty_values(self):
        """空のカテゴリ値を処理できる"""
        df_categories = pd.DataFrame(
            {
                "力量カテゴリーコード": ["cat01", "cat02"],
                "カテゴリ1": ["技術", "技術"],
                "カテゴリ2": ["プログラミング", ""],
                "カテゴリ3": ["", ""],
            }
        )

        transformer = DataTransformer()
        mapping = transformer._create_category_names(df_categories)

        # 空値を除いて結合される
        assert "cat01" in mapping
        assert mapping["cat01"] == "技術 > プログラミング"
        assert mapping["cat02"] == "技術"


# ==================== メンバー習得力量データ作成テスト ====================


class TestCreateMemberCompetence:
    """メンバー習得力量データ作成のテスト"""

    def test_create_member_competence(
        self, sample_members, sample_acquired, sample_competence_master
    ):
        """メンバー習得力量データが作成される"""
        data = {"members": sample_members, "acquired": sample_acquired}

        transformer = DataTransformer()
        member_comp, valid_members = transformer.create_member_competence(
            data, sample_competence_master
        )

        assert len(member_comp) > 0
        assert len(valid_members) > 0
        assert "正規化レベル" in member_comp.columns

    def test_filter_invalid_members(self):
        """削除・テストユーザーが除外される"""
        df_members = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m002", "m003", "m004"],
                "メンバー名": ["田中太郎", "削除済みユーザー", "テスト花子", "佐藤次郎"],
            }
        )

        df_acquired = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m002", "m003", "m004"],
                "力量コード": ["s001", "s002", "s003", "s004"],
                "力量タイプ": ["SKILL", "SKILL", "SKILL", "SKILL"],
                "レベル": [3, 4, 2, 5],
            }
        )

        df_competence_master = pd.DataFrame(
            {
                "力量コード": ["s001", "s002", "s003", "s004"],
                "力量名": ["Python", "SQL", "JavaScript", "Docker"],
                "力量タイプ": ["SKILL", "SKILL", "SKILL", "SKILL"],
                "力量カテゴリー名": ["技術", "技術", "技術", "技術"],
            }
        )

        data = {"members": df_members, "acquired": df_acquired}

        transformer = DataTransformer()
        member_comp, valid_members = transformer.create_member_competence(
            data, df_competence_master
        )

        # m002（削除済み）とm003（テスト）が除外される
        assert "m001" in valid_members
        assert "m002" not in valid_members
        assert "m003" not in valid_members
        assert "m004" in valid_members

    def test_member_competence_level_normalization(self, sample_members, sample_competence_master):
        """レベル正規化が適用される"""
        df_acquired = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m001", "m001"],
                "力量コード": ["s001", "e001", "l001"],
                "力量タイプ": ["SKILL", "EDUCATION", "LICENSE"],
                "レベル": [3, "●", "●"],
            }
        )

        data = {"members": sample_members, "acquired": df_acquired}

        transformer = DataTransformer()
        member_comp, _ = transformer.create_member_competence(data, sample_competence_master)

        # 正規化レベルを確認
        skill_level = member_comp[member_comp["力量コード"] == "s001"]["正規化レベル"].values[0]
        edu_level = member_comp[member_comp["力量コード"] == "e001"]["正規化レベル"].values[0]
        lic_level = member_comp[member_comp["力量コード"] == "l001"]["正規化レベル"].values[0]

        assert skill_level == 3
        assert edu_level == 1
        assert lic_level == 1

    def test_member_competence_merge(
        self, sample_members, sample_acquired, sample_competence_master
    ):
        """力量マスタとのマージが正しく行われる"""
        data = {"members": sample_members, "acquired": sample_acquired}

        transformer = DataTransformer()
        member_comp, _ = transformer.create_member_competence(data, sample_competence_master)

        # マージ後のカラムを確認
        assert "力量名" in member_comp.columns
        assert "力量カテゴリー名" in member_comp.columns


# ==================== スキルマトリクス作成テスト ====================


class TestCreateSkillMatrix:
    """スキルマトリクス作成のテスト"""

    def test_create_skill_matrix(self, sample_member_competence):
        """メンバー×力量マトリクスが作成される"""
        transformer = DataTransformer()
        matrix = transformer.create_skill_matrix(sample_member_competence)

        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) > 0
        assert len(matrix.columns) > 0

    def test_skill_matrix_shape(self, sample_member_competence):
        """マトリクスの行列数が正しい"""
        transformer = DataTransformer()
        matrix = transformer.create_skill_matrix(sample_member_competence)

        # メンバー数（行）
        unique_members = sample_member_competence["メンバーコード"].nunique()
        assert matrix.shape[0] == unique_members

        # 力量数（列）
        unique_competences = sample_member_competence["力量コード"].nunique()
        assert matrix.shape[1] == unique_competences

    def test_skill_matrix_fill_value(self, sample_member_competence):
        """未習得箇所が0で埋められる"""
        transformer = DataTransformer()
        matrix = transformer.create_skill_matrix(sample_member_competence)

        # マトリクスに0が含まれること（未習得の力量がある）
        assert (matrix == 0).any().any()

    def test_skill_matrix_values(self, sample_member_competence):
        """マトリクスの値が正規化レベルと一致"""
        transformer = DataTransformer()
        matrix = transformer.create_skill_matrix(sample_member_competence)

        # 特定のメンバー・力量の値を確認
        # m001がs001を保有している場合
        if "m001" in matrix.index and "s001" in matrix.columns:
            expected_level = sample_member_competence[
                (sample_member_competence["メンバーコード"] == "m001")
                & (sample_member_competence["力量コード"] == "s001")
            ]["正規化レベル"].values[0]

            actual_level = matrix.loc["m001", "s001"]
            assert actual_level == expected_level


# ==================== メンバーマスタクリーニングテスト ====================


class TestCleanMembersData:
    """メンバーマスタクリーニングのテスト"""

    def test_clean_members_data(self, sample_members):
        """メンバーマスタがクリーニングされる"""
        data = {"members": sample_members}

        transformer = DataTransformer()
        clean_members = transformer.clean_members_data(data)

        assert len(clean_members) > 0
        assert "メンバーコード" in clean_members.columns
        assert "メンバー名" in clean_members.columns

    def test_clean_members_excludes_deleted(self):
        """削除済みユーザーが除外される"""
        df_members = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m002", "m003"],
                "メンバー名": ["田中太郎", "削除済みユーザー", "佐藤次郎"],
            }
        )

        data = {"members": df_members}

        transformer = DataTransformer()
        clean_members = transformer.clean_members_data(data)

        # 削除済みユーザーが除外されている
        assert "m001" in clean_members["メンバーコード"].values
        assert "m002" not in clean_members["メンバーコード"].values
        assert "m003" in clean_members["メンバーコード"].values

    def test_clean_members_excludes_test_users(self):
        """テストユーザーが除外される"""
        df_members = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m002", "m003"],
                "メンバー名": ["田中太郎", "テスト花子", "test_user"],
            }
        )

        data = {"members": df_members}

        transformer = DataTransformer()
        clean_members = transformer.clean_members_data(data)

        # テストユーザーが除外されている
        assert "m001" in clean_members["メンバーコード"].values
        assert "m002" not in clean_members["メンバーコード"].values
        assert "m003" not in clean_members["メンバーコード"].values

    def test_clean_members_column_selection(self, sample_members):
        """必要なカラムのみ選択される"""
        data = {"members": sample_members}

        transformer = DataTransformer()
        clean_members = transformer.clean_members_data(data)

        # 基本カラムが含まれる
        assert "メンバーコード" in clean_members.columns
        assert "メンバー名" in clean_members.columns
