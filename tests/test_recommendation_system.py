"""
RecommendationSystemクラスのテスト

推薦システムのファサードインターフェースをテスト
"""

import pytest
import pandas as pd
import os
from skillnote_recommendation.core.recommendation_system import RecommendationSystem


# ==================== フィクスチャ ====================

@pytest.fixture
def temp_output_files(temp_output_dir, sample_members, sample_competence_master,
                      sample_member_competence, sample_similarity):
    """一時出力ファイルを作成"""
    # 必要なCSVファイルを作成
    sample_members.to_csv(
        temp_output_dir / 'members_clean.csv',
        index=False,
        encoding='utf-8-sig'
    )

    sample_competence_master.to_csv(
        temp_output_dir / 'competence_master.csv',
        index=False,
        encoding='utf-8-sig'
    )

    sample_member_competence.to_csv(
        temp_output_dir / 'member_competence.csv',
        index=False,
        encoding='utf-8-sig'
    )

    sample_similarity.to_csv(
        temp_output_dir / 'competence_similarity.csv',
        index=False,
        encoding='utf-8-sig'
    )

    return temp_output_dir


# ==================== 初期化テスト ====================

class TestSystemInitialization:
    """推薦システム初期化のテスト"""

    def test_system_initialization(self, temp_output_files, monkeypatch):
        """システム初期化とデータ読み込み"""
        # Configをモック
        from skillnote_recommendation.core import config

        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))

        assert system is not None
        assert system.df_members is not None
        assert system.df_competence_master is not None
        assert system.df_member_competence is not None
        assert system.df_similarity is not None
        assert system.engine is not None

    def test_initialization_missing_files(self, temp_output_dir):
        """データファイル欠落時に例外"""
        with pytest.raises(FileNotFoundError):
            RecommendationSystem(output_dir=str(temp_output_dir))


# ==================== 会員情報取得テスト ====================

class TestGetMemberInfo:
    """会員情報取得のテスト"""

    def test_get_member_info(self, temp_output_files, monkeypatch):
        """会員情報が取得できる"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        info = system.get_member_info('m001')

        assert info is not None
        assert 'member_code' in info
        assert 'name' in info
        assert 'skill_count' in info
        assert 'education_count' in info
        assert 'license_count' in info

    def test_get_member_info_not_found(self, temp_output_files, monkeypatch):
        """存在しない会員でNone返却"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        info = system.get_member_info('m999')

        assert info is None

    def test_member_info_structure(self, temp_output_files, monkeypatch):
        """返却辞書の構造確認"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        info = system.get_member_info('m001')

        assert info['member_code'] == 'm001'
        assert isinstance(info['name'], str)
        assert isinstance(info['skill_count'], int)
        assert isinstance(info['education_count'], int)
        assert isinstance(info['license_count'], int)

    def test_member_info_competence_counts(self, temp_output_files, monkeypatch):
        """保有力量数の集計が正しい"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        info = system.get_member_info('m001')

        # m001はs001(SKILL), s002(SKILL), e001(EDUCATION)を保有
        assert info['skill_count'] == 2
        assert info['education_count'] == 1
        assert info['license_count'] == 0


# ==================== 力量推薦テスト ====================

class TestRecommendCompetences:
    """力量推薦実行のテスト"""

    def test_recommend_competences(self, temp_output_files, monkeypatch):
        """力量推薦が実行できる"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        recommendations = system.recommend_competences('m001', top_n=5)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

    def test_recommend_with_type_filter(self, temp_output_files, monkeypatch):
        """力量タイプフィルタ適用"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        recommendations = system.recommend_competences(
            'm001',
            competence_type='SKILL',
            top_n=5
        )

        assert all(rec.competence_type == 'SKILL' for rec in recommendations)

    def test_recommend_with_category_filter(self, temp_output_files, monkeypatch):
        """カテゴリフィルタ適用"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        recommendations = system.recommend_competences(
            'm001',
            category_filter='プログラミング',
            top_n=5
        )

        assert all('プログラミング' in rec.category for rec in recommendations)


# ==================== 推薦結果表示テスト ====================

class TestPrintRecommendations:
    """推薦結果表示のテスト"""

    def test_print_recommendations(self, temp_output_files, monkeypatch, capsys):
        """推薦結果が表示される"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        system.print_recommendations('m001', top_n=3)

        captured = capsys.readouterr()

        # 出力に推薦結果が含まれる
        assert '力量推薦結果' in captured.out or '推薦' in captured.out

    def test_print_recommendations_invalid_member(self, temp_output_files,
                                                  monkeypatch, capsys):
        """存在しない会員の場合"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        system.print_recommendations('m999', top_n=5)

        captured = capsys.readouterr()

        # エラーメッセージが出力される
        assert '見つかりません' in captured.out


# ==================== CSV出力テスト ====================

class TestExportRecommendations:
    """CSV出力機能のテスト"""

    def test_export_recommendations(self, temp_output_files, monkeypatch):
        """CSV出力が正常に動作"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        output_file = 'test_recommendations.csv'

        system.export_recommendations('m001', output_file, top_n=5)

        output_path = temp_output_files / output_file
        assert output_path.exists()

    def test_export_file_content(self, temp_output_files, monkeypatch):
        """出力CSVの内容検証"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        output_file = 'test_recommendations.csv'

        system.export_recommendations('m001', output_file, top_n=3)

        output_path = temp_output_files / output_file
        df = pd.read_csv(output_path, encoding='utf-8-sig')

        # データが正しく出力されている
        assert len(df) <= 3
        assert '力量名' in df.columns
        assert '優先度スコア' in df.columns

    def test_export_encoding(self, temp_output_files, monkeypatch):
        """出力エンコーディング確認"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))
        output_file = 'test_recommendations_encoding.csv'

        system.export_recommendations('m001', output_file, top_n=5)

        output_path = temp_output_files / output_file

        # UTF-8-sigで読み込める
        df = pd.read_csv(output_path, encoding='utf-8-sig')
        assert df is not None

    def test_export_no_recommendations(self, temp_output_files, monkeypatch, capsys):
        """推薦結果がない場合"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        # 全ての力量を保有している会員データを作成
        df_members = pd.DataFrame({'メンバーコード': ['m999']})
        df_competence_master = pd.DataFrame({
            '力量コード': ['s001'],
            '力量名': ['Python'],
            '力量タイプ': ['SKILL'],
            '力量カテゴリー名': ['技術']
        })
        df_member_competence = pd.DataFrame({
            'メンバーコード': ['m999'],
            '力量コード': ['s001'],
            '正規化レベル': [3],
            '力量タイプ': ['SKILL'],
            '力量カテゴリー名': ['技術']
        })
        df_similarity = pd.DataFrame(columns=['力量1', '力量2', '類似度'])

        # ファイル作成
        df_members.to_csv(temp_output_files / 'members_clean.csv', index=False, encoding='utf-8-sig')
        df_competence_master.to_csv(temp_output_files / 'competence_master.csv', index=False, encoding='utf-8-sig')
        df_member_competence.to_csv(temp_output_files / 'member_competence.csv', index=False, encoding='utf-8-sig')
        df_similarity.to_csv(temp_output_files / 'competence_similarity.csv', index=False, encoding='utf-8-sig')

        system = RecommendationSystem(output_dir=str(temp_output_files))
        output_file = 'test_empty.csv'

        # 全て習得済みなので推薦結果なし
        system.export_recommendations('m999', output_file, top_n=5)

        captured = capsys.readouterr()

        # メッセージが出力される
        assert '推薦できる力量がありません' in captured.out


# ==================== 統合動作テスト ====================

class TestIntegration:
    """統合動作のテスト"""

    def test_full_workflow(self, temp_output_files, monkeypatch):
        """完全なワークフロー: 初期化→会員情報→推薦→出力"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        # 1. 初期化
        system = RecommendationSystem(output_dir=str(temp_output_files))

        # 2. 会員情報取得
        info = system.get_member_info('m001')
        assert info is not None

        # 3. 推薦実行
        recommendations = system.recommend_competences('m001', top_n=5)
        assert len(recommendations) <= 5

        # 4. CSV出力
        output_file = 'workflow_test.csv'
        system.export_recommendations('m001', output_file, top_n=5)

        output_path = temp_output_files / output_file
        assert output_path.exists()

    def test_multiple_members(self, temp_output_files, monkeypatch):
        """複数の会員に対する推薦"""
        from skillnote_recommendation.core import config
        monkeypatch.setattr(config.Config, 'OUTPUT_DIR', str(temp_output_files))

        system = RecommendationSystem(output_dir=str(temp_output_files))

        # 複数の会員に推薦
        member_codes = ['m001', 'm002', 'm003']
        for member_code in member_codes:
            recommendations = system.recommend_competences(member_code, top_n=3)
            assert isinstance(recommendations, list)
