"""
DataLoaderクラスのテスト

CSVファイルの読み込み、カラム名のクリーニング、データ検証機能をテスト
"""

import pytest
import pandas as pd
from skillnote_recommendation.core.data_loader import DataLoader


# ==================== カラム名クリーニングのテスト ====================

class TestCleanColumnName:
    """カラム名クリーニング機能のテスト"""

    def test_clean_column_name_with_marker(self):
        """マーカー付きカラム名のクリーニング"""
        result = DataLoader.clean_column_name('メンバー名 ###[Member Name]###')
        assert result == 'メンバー名'

    def test_clean_column_name_with_spaces(self):
        """前後の空白を除去"""
        result = DataLoader.clean_column_name('  力量コード  ')
        assert result == '力量コード'

    def test_clean_column_name_with_both(self):
        """マーカーと空白の両方を処理"""
        result = DataLoader.clean_column_name('  レベル ###[Level]###  ')
        assert result == 'レベル'

    def test_clean_column_name_multiple_markers(self):
        """複数のマーカーを含む場合"""
        result = DataLoader.clean_column_name('名前 ###[Name]### ###[Full Name]###')
        assert result == '名前'

    def test_clean_column_name_no_marker(self):
        """マーカーなしのカラム名（そのまま返す）"""
        result = DataLoader.clean_column_name('メンバーコード')
        assert result == 'メンバーコード'

    def test_clean_column_name_empty(self):
        """空文字列"""
        result = DataLoader.clean_column_name('   ')
        assert result == ''


# ==================== CSVファイル読み込みのテスト ====================

class TestLoadCSV:
    """CSV読み込み機能のテスト"""

    def test_load_csv_success(self, temp_data_dir, sample_members):
        """正常なCSVファイルの読み込み"""
        # CSVファイルを作成
        csv_path = temp_data_dir / 'test_members.csv'
        sample_members.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # DataLoaderで読み込み
        loader = DataLoader(data_dir=str(temp_data_dir))
        df = loader.load_csv('test_members.csv')

        assert len(df) == len(sample_members)
        assert 'メンバーコード' in df.columns
        assert 'メンバー名' in df.columns

    def test_load_csv_with_column_cleaning(self, temp_data_dir):
        """カラム名が自動的にクリーニングされる"""
        # マーカー付きカラム名のCSVを作成
        df_dirty = pd.DataFrame({
            'メンバーコード ###[Code]###': ['m001', 'm002'],
            '  メンバー名  ': ['太郎', '花子']
        })
        csv_path = temp_data_dir / 'dirty_columns.csv'
        df_dirty.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 読み込み
        loader = DataLoader(data_dir=str(temp_data_dir))
        df = loader.load_csv('dirty_columns.csv')

        # クリーニング後のカラム名を確認
        assert 'メンバーコード' in df.columns
        assert 'メンバー名' in df.columns
        assert 'メンバーコード ###[Code]###' not in df.columns

    def test_load_csv_file_not_found(self, temp_data_dir):
        """存在しないファイルを読み込むとFileNotFoundError"""
        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_csv('non_existent.csv')

        assert 'non_existent.csv' in str(exc_info.value)

    def test_load_csv_encoding_utf8_sig(self, temp_data_dir):
        """UTF-8-sigエンコーディングで正しく読み込める"""
        # BOM付きUTF-8で保存
        df = pd.DataFrame({
            'メンバーコード': ['m001', 'm002'],
            'メンバー名': ['田中', '鈴木']
        })
        csv_path = temp_data_dir / 'utf8_sig.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 読み込み
        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv('utf8_sig.csv')

        assert len(result) == 2
        assert result.iloc[0]['メンバー名'] == '田中'

    def test_load_csv_custom_data_dir(self, temp_data_dir, sample_members):
        """カスタムデータディレクトリを指定"""
        csv_path = temp_data_dir / 'custom.csv'
        sample_members.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # カスタムディレクトリを指定
        loader = DataLoader(data_dir=str(temp_data_dir))
        df = loader.load_csv('custom.csv')

        assert len(df) == len(sample_members)


# ==================== 全データ読み込みのテスト ====================

class TestLoadAllData:
    """全CSVファイル一括読み込みのテスト"""

    def test_load_all_data_success(self, temp_data_dir, sample_csv_files):
        """全ファイルの正常な読み込み"""
        loader = DataLoader(data_dir=str(temp_data_dir))
        data = loader.load_all_data()

        # 6つのファイルが読み込まれること
        assert len(data) == 6
        assert 'members' in data
        assert 'skills' in data
        assert 'education' in data
        assert 'license' in data
        assert 'categories' in data
        assert 'acquired' in data

        # 各DataFrameが空でないこと
        for key, df in data.items():
            assert len(df) > 0, f"{key} のデータが空です"

    def test_load_all_data_missing_file(self, temp_data_dir):
        """ファイルが1つでも欠けていたら例外"""
        # membersファイルだけ作成
        df = pd.DataFrame({'メンバーコード': ['m001']})
        (temp_data_dir / 'member_skillnote.csv').write_text(df.to_csv(index=False))

        loader = DataLoader(data_dir=str(temp_data_dir))

        with pytest.raises(FileNotFoundError):
            loader.load_all_data()

    def test_load_all_data_returns_dict(self, temp_data_dir, sample_csv_files):
        """辞書形式で返されること"""
        loader = DataLoader(data_dir=str(temp_data_dir))
        data = loader.load_all_data()

        assert isinstance(data, dict)
        for key, value in data.items():
            assert isinstance(value, pd.DataFrame)


# ==================== データ検証のテスト ====================

class TestValidateData:
    """データ整合性検証のテスト"""

    def test_validate_data_success(self, temp_data_dir, sample_csv_files):
        """正常データの検証が成功"""
        loader = DataLoader(data_dir=str(temp_data_dir))
        data = loader.load_all_data()

        result = loader.validate_data(data)

        assert result is True

    def test_validate_data_missing_members_columns(self, sample_csv_files):
        """membersの必須カラムが欠けている場合"""
        # メンバーコードが欠けたDataFrame
        data = {
            'members': pd.DataFrame({'メンバー名': ['太郎']}),
            'skills': pd.DataFrame({'力量コード': ['s001'], '力量名': ['Python']}),
            'education': pd.DataFrame({'力量コード': ['e001'], '力量名': ['研修']}),
            'license': pd.DataFrame({'力量コード': ['l001'], '力量名': ['資格']}),
            'categories': pd.DataFrame({'力量カテゴリーコード': ['cat01']}),
            'acquired': pd.DataFrame({
                'メンバーコード': ['m001'],
                '力量コード': ['s001'],
                '力量タイプ': ['SKILL'],
                'レベル': [3]
            })
        }

        loader = DataLoader()
        result = loader.validate_data(data)

        assert result is False

    def test_validate_data_missing_acquired_columns(self):
        """acquiredの必須カラムが欠けている場合"""
        data = {
            'members': pd.DataFrame({'メンバーコード': ['m001'], 'メンバー名': ['太郎']}),
            'skills': pd.DataFrame({'力量コード': ['s001'], '力量名': ['Python']}),
            'education': pd.DataFrame({'力量コード': ['e001'], '力量名': ['研修']}),
            'license': pd.DataFrame({'力量コード': ['l001'], '力量名': ['資格']}),
            'categories': pd.DataFrame({'力量カテゴリーコード': ['cat01']}),
            'acquired': pd.DataFrame({
                'メンバーコード': ['m001'],
                '力量コード': ['s001']
                # '力量タイプ', 'レベル' が欠けている
            })
        }

        loader = DataLoader()
        result = loader.validate_data(data)

        assert result is False

    def test_validate_data_all_files_valid(self, temp_data_dir, sample_csv_files):
        """全ファイルの必須カラムが揃っている"""
        loader = DataLoader(data_dir=str(temp_data_dir))
        data = loader.load_all_data()

        # 各ファイルの必須カラムを確認
        assert 'メンバーコード' in data['members'].columns
        assert 'メンバー名' in data['members'].columns
        assert '力量コード' in data['skills'].columns
        assert '力量名' in data['skills'].columns

        result = loader.validate_data(data)
        assert result is True


# ==================== DataLoader初期化のテスト ====================

class TestDataLoaderInit:
    """DataLoader初期化のテスト"""

    def test_init_default_data_dir(self):
        """デフォルトのデータディレクトリが設定される"""
        loader = DataLoader()
        assert loader.data_dir is not None
        assert 'data' in loader.data_dir

    def test_init_custom_data_dir(self, temp_data_dir):
        """カスタムデータディレクトリが設定される"""
        custom_dir = str(temp_data_dir)
        loader = DataLoader(data_dir=custom_dir)
        assert loader.data_dir == custom_dir


# ==================== エッジケーステスト ====================

class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_csv_file(self, temp_data_dir):
        """空のCSVファイル（pandasはEmptyDataErrorを発生）"""
        csv_path = temp_data_dir / 'empty.csv'
        pd.DataFrame().to_csv(csv_path, index=False, encoding='utf-8-sig')

        loader = DataLoader(data_dir=str(temp_data_dir))

        # 空のCSVファイルはpandasでEmptyDataErrorを発生させる
        with pytest.raises(pd.errors.EmptyDataError):
            loader.load_csv('empty.csv')

    def test_csv_with_single_row(self, temp_data_dir):
        """1行だけのCSVファイル"""
        df = pd.DataFrame({'メンバーコード': ['m001'], 'メンバー名': ['太郎']})
        csv_path = temp_data_dir / 'single_row.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv('single_row.csv')

        assert len(result) == 1
        assert result.iloc[0]['メンバーコード'] == 'm001'

    def test_csv_with_special_characters(self, temp_data_dir):
        """特殊文字を含むデータ"""
        df = pd.DataFrame({
            'メンバー名': ['田中,太郎', '鈴木"花子"', '佐藤\n次郎']
        })
        csv_path = temp_data_dir / 'special_chars.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        loader = DataLoader(data_dir=str(temp_data_dir))
        result = loader.load_csv('special_chars.csv')

        assert len(result) == 3
        # CSVエスケープが正しく処理される
        assert '田中,太郎' in result['メンバー名'].values
