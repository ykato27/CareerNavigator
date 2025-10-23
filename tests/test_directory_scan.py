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
        members_dir = temp_data_dir / 'members'
        members_dir.mkdir()

        # 複数のCSVファイルを作成
        df1 = pd.DataFrame({
            'メンバーコード': ['m001', 'm002'],
            'メンバー名': ['田中太郎', '鈴木花子']
        })
        df2 = pd.DataFrame({
            'メンバーコード': ['m003', 'm004'],
            'メンバー名': ['佐藤次郎', '高橋美咲']
        })
        df3 = pd.DataFrame({
            'メンバーコード': ['m005'],
            'メンバー名': ['伊藤健一']
        })

        df1.to_csv(members_dir / 'member_1.csv', index=False, encoding='utf-8-sig')
        df2.to_csv(members_dir / 'member_2.csv', index=False, encoding='utf-8-sig')
        df3.to_csv(members_dir / 'member_3.csv', index=False, encoding='utf-8-sig')

        # DataLoaderで読み込み
        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory('members')

        # 3ファイルが結合されて5行になる
        assert len(result) == 5
        assert 'm001' in result['メンバーコード'].values
        assert 'm005' in result['メンバーコード'].values

    def test_load_csv_from_directory_single_file(self, temp_data_dir):
        """ディレクトリ内に1ファイルだけの場合"""
        members_dir = temp_data_dir / 'members'
        members_dir.mkdir()

        df = pd.DataFrame({
            'メンバーコード': ['m001', 'm002'],
            'メンバー名': ['田中太郎', '鈴木花子']
        })
        df.to_csv(members_dir / 'member.csv', index=False, encoding='utf-8-sig')

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory('members')

        assert len(result) == 2

    def test_load_csv_from_directory_not_exists(self, temp_data_dir):
        """存在しないディレクトリでFileNotFoundError"""
        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(FileNotFoundError):
            loader.load_csv_from_directory('non_existent')

    def test_load_csv_from_directory_empty(self, temp_data_dir):
        """CSVファイルがないディレクトリでFileNotFoundError"""
        empty_dir = temp_data_dir / 'empty'
        empty_dir.mkdir()

        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_csv_from_directory('empty')

        assert 'CSVファイルが見つかりません' in str(exc_info.value)

    def test_load_csv_from_directory_sorted(self, temp_data_dir):
        """複数ファイルがソート順で読み込まれる"""
        members_dir = temp_data_dir / 'members'
        members_dir.mkdir()

        # 意図的に逆順で作成
        df3 = pd.DataFrame({'メンバーコード': ['m003']})
        df1 = pd.DataFrame({'メンバーコード': ['m001']})
        df2 = pd.DataFrame({'メンバーコード': ['m002']})

        df3.to_csv(members_dir / 'member_3.csv', index=False, encoding='utf-8-sig')
        df1.to_csv(members_dir / 'member_1.csv', index=False, encoding='utf-8-sig')
        df2.to_csv(members_dir / 'member_2.csv', index=False, encoding='utf-8-sig')

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory('members')

        # ソートされているので member_1, member_2, member_3 の順
        assert result['メンバーコード'].tolist() == ['m001', 'm002', 'm003']


# ==================== 柔軟な読み込みテスト ====================

class TestLoadDataFlexible:
    """ディレクトリまたは単一ファイルの柔軟な読み込みテスト"""

    def test_load_data_flexible_directory(self, temp_data_dir, monkeypatch):
        """ディレクトリが存在する場合はディレクトリから読み込む"""
        from skillnote_recommendation.core import config

        # ディレクトリ作成
        members_dir = temp_data_dir / 'members'
        members_dir.mkdir()

        df1 = pd.DataFrame({'メンバーコード': ['m001', 'm002']})
        df2 = pd.DataFrame({'メンバーコード': ['m003']})

        df1.to_csv(members_dir / 'member_1.csv', index=False, encoding='utf-8-sig')
        df2.to_csv(members_dir / 'member_2.csv', index=False, encoding='utf-8-sig')

        monkeypatch.setattr(config.Config, 'DATA_DIR', str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_data_flexible('members')

        assert len(result) == 3

    def test_load_data_flexible_single_file(self, temp_data_dir, monkeypatch):
        """ディレクトリがない場合は単一ファイルから読み込む"""
        from skillnote_recommendation.core import config

        # 単一ファイル作成
        df = pd.DataFrame({'メンバーコード': ['m001', 'm002']})
        df.to_csv(temp_data_dir / 'member_skillnote.csv', index=False, encoding='utf-8-sig')

        monkeypatch.setattr(config.Config, 'DATA_DIR', str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_data_flexible('members')

        assert len(result) == 2

    def test_load_data_flexible_directory_priority(self, temp_data_dir, monkeypatch):
        """ディレクトリと単一ファイルの両方がある場合はディレクトリ優先"""
        from skillnote_recommendation.core import config

        # ディレクトリ作成（3行）
        members_dir = temp_data_dir / 'members'
        members_dir.mkdir()
        df_dir = pd.DataFrame({'メンバーコード': ['m001', 'm002', 'm003']})
        df_dir.to_csv(members_dir / 'member.csv', index=False, encoding='utf-8-sig')

        # 単一ファイル作成（2行）
        df_file = pd.DataFrame({'メンバーコード': ['m004', 'm005']})
        df_file.to_csv(temp_data_dir / 'member_skillnote.csv', index=False, encoding='utf-8-sig')

        monkeypatch.setattr(config.Config, 'DATA_DIR', str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_data_flexible('members')

        # ディレクトリが優先されるので3行
        assert len(result) == 3

    def test_load_data_flexible_not_found(self, temp_data_dir, monkeypatch):
        """ディレクトリもファイルもない場合はFileNotFoundError"""
        from skillnote_recommendation.core import config

        monkeypatch.setattr(config.Config, 'DATA_DIR', str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_data_flexible('members')

        assert 'members' in str(exc_info.value)


# ==================== load_all_data拡張テスト ====================

class TestLoadAllDataWithDirectories:
    """ディレクトリスキャン対応後のload_all_dataテスト"""

    def test_load_all_data_with_directories(self, temp_data_dir, sample_csv_files, monkeypatch):
        """既存の単一ファイル方式で動作することを確認"""
        from skillnote_recommendation.core import config

        monkeypatch.setattr(config.Config, 'DATA_DIR', str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))
        data = loader.load_all_data()

        # 6種類のデータが読み込まれる
        assert len(data) == 6
        assert 'members' in data
        assert 'skills' in data

    def test_load_all_data_mixed_structure(self, temp_data_dir, monkeypatch):
        """ディレクトリと単一ファイルの混在"""
        from skillnote_recommendation.core import config

        # membersはディレクトリ
        members_dir = temp_data_dir / 'members'
        members_dir.mkdir()
        df1 = pd.DataFrame({'メンバーコード': ['m001', 'm002'], 'メンバー名': ['太郎', '花子']})
        df2 = pd.DataFrame({'メンバーコード': ['m003'], 'メンバー名': ['次郎']})
        df1.to_csv(members_dir / 'member_1.csv', index=False, encoding='utf-8-sig')
        df2.to_csv(members_dir / 'member_2.csv', index=False, encoding='utf-8-sig')

        # 他は単一ファイル
        pd.DataFrame({
            'メンバーコード': ['m001'],
            '力量コード': ['s001'],
            '力量タイプ': ['SKILL'],
            'レベル': [3]
        }).to_csv(temp_data_dir / 'acquiredCompetenceLevel.csv', index=False, encoding='utf-8-sig')

        pd.DataFrame({'力量コード': ['s001'], '力量名': ['Python']}).to_csv(
            temp_data_dir / 'skill_skillnote.csv', index=False, encoding='utf-8-sig')

        pd.DataFrame({'力量コード': ['e001'], '力量名': ['研修']}).to_csv(
            temp_data_dir / 'education_skillnote.csv', index=False, encoding='utf-8-sig')

        pd.DataFrame({'力量コード': ['l001'], '力量名': ['資格']}).to_csv(
            temp_data_dir / 'license_skillnote.csv', index=False, encoding='utf-8-sig')

        pd.DataFrame({'力量カテゴリーコード': ['cat01']}).to_csv(
            temp_data_dir / 'competence_category_skillnote.csv', index=False, encoding='utf-8-sig')

        monkeypatch.setattr(config.Config, 'DATA_DIR', str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))
        data = loader.load_all_data()

        # membersは3行（ディレクトリから2ファイル結合）
        assert len(data['members']) == 3
        assert 'm001' in data['members']['メンバーコード'].values
        assert 'm003' in data['members']['メンバーコード'].values


# ==================== 統合テスト ====================

class TestDirectoryScanIntegration:
    """ディレクトリスキャン機能の統合テスト"""

    def test_real_world_scenario(self, temp_data_dir, monkeypatch):
        """実際の使用シナリオ：部署ごとにファイルが分かれている"""
        from skillnote_recommendation.core import config

        # 部署ごとのメンバーファイル
        members_dir = temp_data_dir / 'members'
        members_dir.mkdir()

        df_dept_a = pd.DataFrame({
            'メンバーコード': ['m001', 'm002'],
            'メンバー名': ['部署A 太郎', '部署A 花子']
        })
        df_dept_b = pd.DataFrame({
            'メンバーコード': ['m003', 'm004'],
            'メンバー名': ['部署B 次郎', '部署B 美咲']
        })
        df_dept_c = pd.DataFrame({
            'メンバーコード': ['m005'],
            'メンバー名': ['部署C 健一']
        })

        df_dept_a.to_csv(members_dir / 'members_dept_a.csv', index=False, encoding='utf-8-sig')
        df_dept_b.to_csv(members_dir / 'members_dept_b.csv', index=False, encoding='utf-8-sig')
        df_dept_c.to_csv(members_dir / 'members_dept_c.csv', index=False, encoding='utf-8-sig')

        monkeypatch.setattr(config.Config, 'DATA_DIR', str(temp_data_dir))

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv_from_directory('members')

        # 3部署のデータが結合されて5名
        assert len(result) == 5
        assert '部署A' in result['メンバー名'].values[0]
        assert '部署B' in result['メンバー名'].values[2]
        assert '部署C' in result['メンバー名'].values[4]
