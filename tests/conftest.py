"""
共通テストフィクスチャ

全テストで共有するフィクスチャとヘルパー関数を定義
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path


# ==================== テストデータフィクスチャ ====================


@pytest.fixture
def sample_members():
    """サンプルメンバーデータ（最小データセット）"""
    return pd.DataFrame(
        {
            "メンバーコード": ["m001", "m002", "m003", "m004", "m005"],
            "メンバー名": ["田中太郎", "鈴木花子", "佐藤次郎", "高橋美咲", "伊藤健一"],
            "よみがな": [
                "たなかたろう",
                "すずきはなこ",
                "さとうじろう",
                "たかはしみさき",
                "いとうけんいち",
            ],
            "生年月日": ["1990-04-15", "1988-07-22", "1992-01-10", "1995-09-05", "1987-11-30"],
            "性別": ["男性", "女性", "男性", "女性", "男性"],
            "入社年月日": ["2015-04-01", "2013-04-01", "2018-04-01", "2020-04-01", "2012-04-01"],
            "社員区分": ["正社員", "正社員", "正社員", "正社員", "正社員"],
            "役職": ["主任", "係長", "スタッフ", "スタッフ", "課長"],
            "職能・等級": ["3等級", "4等級", "2等級", "2等級", "5等級"],
        }
    )


@pytest.fixture
def sample_skills():
    """サンプルスキル力量マスタ"""
    return pd.DataFrame(
        {
            "力量コード": ["s001", "s002", "s003", "s004", "s005"],
            "力量名": ["Python", "SQL", "JavaScript", "Docker", "Git"],
            "力量カテゴリーコード": ["cat01", "cat02", "cat01", "cat03", "cat03"],
            "概要": [
                "Pythonプログラミング",
                "SQLデータベース",
                "JavaScriptプログラミング",
                "コンテナ技術",
                "バージョン管理",
            ],
        }
    )


@pytest.fixture
def sample_education():
    """サンプル教育力量マスタ"""
    return pd.DataFrame(
        {
            "力量コード": ["e001", "e002", "e003"],
            "力量名": ["AWS研修", "アジャイル開発研修", "セキュリティ基礎研修"],
            "力量カテゴリーコード": ["cat04", "cat05", "cat06"],
            "概要": [
                "AWSクラウドサービス研修",
                "アジャイル開発手法の研修",
                "セキュリティ基礎知識研修",
            ],
        }
    )


@pytest.fixture
def sample_license():
    """サンプル資格力量マスタ"""
    return pd.DataFrame(
        {
            "力量コード": ["l001", "l002", "l003"],
            "力量名": ["基本情報技術者", "応用情報技術者", "AWS認定ソリューションアーキテクト"],
            "力量カテゴリーコード": ["cat07", "cat07", "cat04"],
            "概要": ["基本情報技術者試験", "応用情報技術者試験", "AWS認定資格"],
        }
    )


@pytest.fixture
def sample_categories():
    """サンプルカテゴリマスタ"""
    return pd.DataFrame(
        {
            "力量カテゴリーコード": ["cat01", "cat02", "cat03", "cat04", "cat05", "cat06", "cat07"],
            "カテゴリ1": ["技術", "技術", "技術", "技術", "技術", "技術", "資格"],
            "カテゴリ2": [
                "プログラミング",
                "データベース",
                "インフラ",
                "クラウド",
                "開発手法",
                "セキュリティ",
                "IT資格",
            ],
            "カテゴリ3": ["", "", "", "", "", "", ""],
        }
    )


@pytest.fixture
def sample_acquired():
    """サンプル習得力量データ"""
    return pd.DataFrame(
        {
            "メンバーコード": [
                "m001",
                "m001",
                "m001",
                "m002",
                "m002",
                "m003",
                "m003",
                "m004",
                "m005",
            ],
            "力量コード": ["s001", "s002", "e001", "s001", "s003", "s002", "l001", "s001", "s004"],
            "力量タイプ": [
                "SKILL",
                "SKILL",
                "EDUCATION",
                "SKILL",
                "SKILL",
                "SKILL",
                "LICENSE",
                "SKILL",
                "SKILL",
            ],
            "レベル": [3, 4, "●", 2, 3, 5, "●", 4, 3],
        }
    )


@pytest.fixture
def sample_competence_master():
    """サンプル統合力量マスタ"""
    return pd.DataFrame(
        {
            "力量コード": ["s001", "s002", "s003", "e001", "l001"],
            "力量名": ["Python", "SQL", "JavaScript", "AWS研修", "基本情報技術者"],
            "力量タイプ": ["SKILL", "SKILL", "SKILL", "EDUCATION", "LICENSE"],
            "力量カテゴリー名": [
                "技術 > プログラミング",
                "技術 > データベース",
                "技術 > プログラミング",
                "技術 > クラウド",
                "資格 > IT資格",
            ],
            "レベル範囲": ["1-5", "1-5", "1-5", "●", "●"],
        }
    )


@pytest.fixture
def sample_member_competence():
    """サンプルメンバー習得力量データ（正規化済み）"""
    return pd.DataFrame(
        {
            "メンバーコード": [
                "m001",
                "m001",
                "m001",
                "m002",
                "m002",
                "m003",
                "m003",
                "m004",
                "m005",
            ],
            "力量コード": ["s001", "s002", "e001", "s001", "s003", "s002", "l001", "s001", "s004"],
            "力量タイプ": [
                "SKILL",
                "SKILL",
                "EDUCATION",
                "SKILL",
                "SKILL",
                "SKILL",
                "LICENSE",
                "SKILL",
                "SKILL",
            ],
            "正規化レベル": [3, 4, 1, 2, 3, 5, 1, 4, 3],
            "力量名": [
                "Python",
                "SQL",
                "AWS研修",
                "Python",
                "JavaScript",
                "SQL",
                "基本情報技術者",
                "Python",
                "Docker",
            ],
            "力量カテゴリー名": [
                "技術 > プログラミング",
                "技術 > データベース",
                "技術 > クラウド",
                "技術 > プログラミング",
                "技術 > プログラミング",
                "技術 > データベース",
                "資格 > IT資格",
                "技術 > プログラミング",
                "技術 > インフラ",
            ],
        }
    )


@pytest.fixture
def sample_similarity():
    """サンプル類似度データ"""
    return pd.DataFrame(
        {
            "力量1": ["s001", "s001", "s002"],
            "力量2": ["s002", "s003", "s003"],
            "類似度": [0.35, 0.42, 0.28],
        }
    )


# ==================== 一時ディレクトリフィクスチャ ====================


@pytest.fixture
def temp_data_dir(tmp_path):
    """一時データディレクトリを作成"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """一時出力ディレクトリを作成"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# ==================== CSVファイルフィクスチャ ====================


@pytest.fixture
def sample_csv_files(
    temp_data_dir,
    sample_members,
    sample_skills,
    sample_education,
    sample_license,
    sample_categories,
    sample_acquired,
):
    """サンプルCSVファイルをディレクトリ構造で作成"""
    files = {}

    # 各ディレクトリを作成し、DataFrameをCSVとして保存
    members_dir = temp_data_dir / "members"
    members_dir.mkdir()
    files["members"] = members_dir / "members.csv"
    sample_members.to_csv(files["members"], index=False, encoding="utf-8-sig")

    skills_dir = temp_data_dir / "skills"
    skills_dir.mkdir()
    files["skills"] = skills_dir / "skills.csv"
    sample_skills.to_csv(files["skills"], index=False, encoding="utf-8-sig")

    education_dir = temp_data_dir / "education"
    education_dir.mkdir()
    files["education"] = education_dir / "education.csv"
    sample_education.to_csv(files["education"], index=False, encoding="utf-8-sig")

    license_dir = temp_data_dir / "license"
    license_dir.mkdir()
    files["license"] = license_dir / "license.csv"
    sample_license.to_csv(files["license"], index=False, encoding="utf-8-sig")

    categories_dir = temp_data_dir / "categories"
    categories_dir.mkdir()
    files["categories"] = categories_dir / "categories.csv"
    sample_categories.to_csv(files["categories"], index=False, encoding="utf-8-sig")

    acquired_dir = temp_data_dir / "acquired"
    acquired_dir.mkdir()
    files["acquired"] = acquired_dir / "acquired.csv"
    sample_acquired.to_csv(files["acquired"], index=False, encoding="utf-8-sig")

    return files


# ==================== ヘルパー関数 ====================


def create_csv_with_dirty_columns(file_path, data_dict):
    """
    カラム名にマーカーが付いたCSVファイルを作成

    Args:
        file_path: 出力ファイルパス
        data_dict: データ辞書（カラム名: データリスト）
    """
    df = pd.DataFrame(data_dict)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")


def assert_dataframe_equal(df1, df2, check_like=True):
    """
    2つのDataFrameが等しいことをアサート（カラム順序無視オプション付き）

    Args:
        df1: DataFrame 1
        df2: DataFrame 2
        check_like: カラム順序を無視するか（デフォルト: True）
    """
    pd.testing.assert_frame_equal(df1, df2, check_like=check_like)


# ==================== マーカー ====================


def pytest_configure(config):
    """カスタムマーカーを登録"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
