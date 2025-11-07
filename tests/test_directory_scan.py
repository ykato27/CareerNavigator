"""
DataLoaderのディレクトリスキャン機能テスト

複数CSVファイルの読み込みと結合機能をテスト
"""

import pytest
import pandas as pd
from skillnote_recommendation.core.data_loader import DataLoader


# ==================== ディレクトリスキャンテスト ====================


class TestLoadCSVFromDirectory:
    """ディレクトリからの複数CSV読み込みテスト"""

    def test_load_csv_from_directory_multiple_files(self, temp_data_dir):
        """ディレクトリ内の複数CSVファイルを読み込んで結合"""
        # サブディレクトリ作成
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        # 複数のCSVファイルを作成
        df1 = pd.DataFrame(
            {"メンバーコード": ["m001", "m002"], "メンバー名": ["田中太郎", "鈴木花子"]}
        )
        df2 = pd.DataFrame(
            {"メンバーコード": ["m003", "m004"], "メンバー名": ["佐藤次郎", "高橋美咲"]}
        )
        df3 = pd.DataFrame({"メンバーコード": ["m005"], "メンバー名": ["伊藤健一"]})

        df1.to_csv(members_dir / "member_1.csv", index=False, encoding="utf-8-sig")
        df2.to_csv(members_dir / "member_2.csv", index=False, encoding="utf-8-sig")
        df3.to_csv(members_dir / "member_3.csv", index=False, encoding="utf-8-sig")

        # DataLoaderで読み込み
        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory("members")

        # 3ファイルが結合されて5行になる
        assert len(result) == 5
        assert "m001" in result["メンバーコード"].values
        assert "m005" in result["メンバーコード"].values

    def test_load_csv_from_directory_single_file(self, temp_data_dir):
        """ディレクトリ内に1ファイルだけの場合"""
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        df = pd.DataFrame(
            {"メンバーコード": ["m001", "m002"], "メンバー名": ["田中太郎", "鈴木花子"]}
        )
        df.to_csv(members_dir / "member.csv", index=False, encoding="utf-8-sig")

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory("members")

        assert len(result) == 2

    def test_load_csv_from_directory_not_exists(self, temp_data_dir):
        """存在しないディレクトリでFileNotFoundError"""
        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(FileNotFoundError):
            loader.load_csv_from_directory("non_existent")

    def test_load_csv_from_directory_empty(self, temp_data_dir):
        """CSVファイルがないディレクトリでFileNotFoundError"""
        empty_dir = temp_data_dir / "empty"
        empty_dir.mkdir()

        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_csv_from_directory("empty")

        assert "CSVファイルが見つかりません" in str(exc_info.value)

    def test_load_csv_from_directory_sorted(self, temp_data_dir):
        """複数ファイルがソート順で読み込まれる"""
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        # 意図的に逆順で作成
        df3 = pd.DataFrame({"メンバーコード": ["m003"]})
        df1 = pd.DataFrame({"メンバーコード": ["m001"]})
        df2 = pd.DataFrame({"メンバーコード": ["m002"]})

        df3.to_csv(members_dir / "member_3.csv", index=False, encoding="utf-8-sig")
        df1.to_csv(members_dir / "member_1.csv", index=False, encoding="utf-8-sig")
        df2.to_csv(members_dir / "member_2.csv", index=False, encoding="utf-8-sig")

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory("members")

        # ソートされているので member_1, member_2, member_3 の順
        assert result["メンバーコード"].tolist() == ["m001", "m002", "m003"]


# ==================== カラム整合性チェックテスト ====================


class TestColumnConsistencyCheck:
    """カラム構造整合性チェックのテスト"""

    def test_inconsistent_columns_raises_error(self, temp_data_dir):
        """カラム構造が異なるファイルがあるとValueError"""
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        # 正常なファイル
        df1 = pd.DataFrame(
            {"メンバーコード": ["m001", "m002"], "メンバー名": ["田中太郎", "鈴木花子"]}
        )
        # カラム構造が異なるファイル
        df2 = pd.DataFrame(
            {
                "メンバーコード": ["m003"],
                "名前": ["佐藤次郎"],  # カラム名が違う
                "部署": ["営業部"],  # 余分なカラム
            }
        )

        df1.to_csv(members_dir / "member_1.csv", index=False, encoding="utf-8-sig")
        df2.to_csv(members_dir / "member_2.csv", index=False, encoding="utf-8-sig")

        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(ValueError) as exc_info:
            loader.load_csv_from_directory("members")

        # エラーメッセージの確認
        error_msg = str(exc_info.value)
        assert "カラム構造が一致しません" in error_msg
        assert "member_1.csv" in error_msg
        assert "member_2.csv" in error_msg

    def test_missing_columns_detected(self, temp_data_dir):
        """不足しているカラムが検出される"""
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        df1 = pd.DataFrame(
            {"メンバーコード": ["m001"], "メンバー名": ["田中太郎"], "部署": ["開発部"]}
        )
        df2 = pd.DataFrame(
            {
                "メンバーコード": ["m002"],
                "メンバー名": ["鈴木花子"],
                # '部署'カラムが欠けている
            }
        )

        df1.to_csv(members_dir / "member_1.csv", index=False, encoding="utf-8-sig")
        df2.to_csv(members_dir / "member_2.csv", index=False, encoding="utf-8-sig")

        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(ValueError) as exc_info:
            loader.load_csv_from_directory("members")

        error_msg = str(exc_info.value)
        assert "不足カラム" in error_msg
        assert "部署" in error_msg

    def test_extra_columns_detected(self, temp_data_dir):
        """余分なカラムが検出される"""
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        df1 = pd.DataFrame({"メンバーコード": ["m001"], "メンバー名": ["田中太郎"]})
        df2 = pd.DataFrame(
            {
                "メンバーコード": ["m002"],
                "メンバー名": ["鈴木花子"],
                "余分カラム": ["データ"],  # 余分なカラム
            }
        )

        df1.to_csv(members_dir / "member_1.csv", index=False, encoding="utf-8-sig")
        df2.to_csv(members_dir / "member_2.csv", index=False, encoding="utf-8-sig")

        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(ValueError) as exc_info:
            loader.load_csv_from_directory("members")

        error_msg = str(exc_info.value)
        assert "余分なカラム" in error_msg
        assert "余分カラム" in error_msg

    def test_all_files_same_structure_ok(self, temp_data_dir):
        """全ファイルが同じ構造なら正常に結合される"""
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        # 全て同じカラム構造
        df1 = pd.DataFrame(
            {"メンバーコード": ["m001", "m002"], "メンバー名": ["田中太郎", "鈴木花子"]}
        )
        df2 = pd.DataFrame({"メンバーコード": ["m003"], "メンバー名": ["佐藤次郎"]})
        df3 = pd.DataFrame({"メンバーコード": ["m004"], "メンバー名": ["高橋美咲"]})

        df1.to_csv(members_dir / "member_1.csv", index=False, encoding="utf-8-sig")
        df2.to_csv(members_dir / "member_2.csv", index=False, encoding="utf-8-sig")
        df3.to_csv(members_dir / "member_3.csv", index=False, encoding="utf-8-sig")

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory("members")

        # 正常に結合される
        assert len(result) == 4
        assert set(result.columns) == {"メンバーコード", "メンバー名"}


class TestDirectoryScanIntegration:
    """ディレクトリスキャン機能の統合テスト"""

    def test_real_world_scenario(self, temp_data_dir, monkeypatch):
        """実際の使用シナリオ：部署ごとにファイルが分かれている"""
        from skillnote_recommendation.core import config

        # 部署ごとのメンバーファイル
        members_dir = temp_data_dir / "members"
        members_dir.mkdir()

        df_dept_a = pd.DataFrame(
            {"メンバーコード": ["m001", "m002"], "メンバー名": ["部署A 太郎", "部署A 花子"]}
        )
        df_dept_b = pd.DataFrame(
            {"メンバーコード": ["m003", "m004"], "メンバー名": ["部署B 次郎", "部署B 美咲"]}
        )
        df_dept_c = pd.DataFrame({"メンバーコード": ["m005"], "メンバー名": ["部署C 健一"]})

        df_dept_a.to_csv(members_dir / "members_dept_a.csv", index=False, encoding="utf-8-sig")
        df_dept_b.to_csv(members_dir / "members_dept_b.csv", index=False, encoding="utf-8-sig")
        df_dept_c.to_csv(members_dir / "members_dept_c.csv", index=False, encoding="utf-8-sig")

        monkeypatch.setattr(config.Config, "DATA_DIR", str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory("members")

        # 3部署のデータが結合されて5名
        assert len(result) == 5
        assert "部署A" in result["メンバー名"].values[0]
        assert "部署B" in result["メンバー名"].values[2]
        assert "部署C" in result["メンバー名"].values[4]
