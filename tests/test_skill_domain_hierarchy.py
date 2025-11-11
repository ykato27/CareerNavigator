"""
SkillDomainHierarchyのテスト

スキル領域階層構造の定義とキーワードマッチングのテスト
"""

import pytest
import pandas as pd
import numpy as np

from skillnote_recommendation.ml.skill_domain_hierarchy import SkillDomainHierarchy


@pytest.fixture
def sample_competence_master():
    """テスト用の力量マスタ"""
    return pd.DataFrame({
        '力量コード': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008'],
        '力量名': [
            'Python基礎',
            'Java基礎',
            'Web開発',
            'API開発',
            'システム設計',
            'SQL基礎',
            'DB設計',
            'Excel分析',
        ],
    })


class TestSkillDomainHierarchyInit:
    """初期化のテスト"""

    def test_initialization(self, sample_competence_master):
        """初期化が正常に行われるかテスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        assert hierarchy is not None
        assert len(hierarchy.hierarchy) > 0
        assert len(hierarchy.competence_classification) > 0

    def test_default_hierarchy_structure(self, sample_competence_master):
        """デフォルト階層構造が正しく定義されているかテスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # 5つのドメインが定義されている
        assert 'プログラミング' in hierarchy.hierarchy
        assert 'データベース' in hierarchy.hierarchy
        assert 'データ分析' in hierarchy.hierarchy
        assert 'マネジメント' in hierarchy.hierarchy
        assert 'コミュニケーション' in hierarchy.hierarchy

        # 各ドメインに3レベルが定義されている
        for domain in hierarchy.hierarchy.keys():
            assert 'level_1' in hierarchy.hierarchy[domain]
            assert 'level_2' in hierarchy.hierarchy[domain]
            assert 'level_3' in hierarchy.hierarchy[domain]


class TestCompetenceClassification:
    """力量分類のテスト"""

    def test_keyword_matching(self, sample_competence_master):
        """キーワードマッチングが機能するかテスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # Python基礎 → プログラミング Level 1
        assert hierarchy.get_domain('C001') == 'プログラミング'
        assert hierarchy.get_level('C001') == 1

        # Web開発 → プログラミング Level 2
        assert hierarchy.get_domain('C003') == 'プログラミング'
        assert hierarchy.get_level('C003') == 2

        # システム設計 → プログラミング Level 3
        assert hierarchy.get_domain('C005') == 'プログラミング'
        assert hierarchy.get_level('C005') == 3

    def test_database_classification(self, sample_competence_master):
        """データベース関連の分類テスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # SQL基礎 → データベース Level 1
        assert hierarchy.get_domain('C006') == 'データベース'
        assert hierarchy.get_level('C006') == 1

        # DB設計 → データベース Level 3
        assert hierarchy.get_domain('C007') == 'データベース'
        assert hierarchy.get_level('C007') == 3

    def test_data_analysis_classification(self, sample_competence_master):
        """データ分析関連の分類テスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # Excel分析 → データ分析 Level 1
        assert hierarchy.get_domain('C008') == 'データ分析'
        assert hierarchy.get_level('C008') == 1

    def test_unclassified_competence(self):
        """分類できない力量のテスト"""
        competence_master = pd.DataFrame({
            '力量コード': ['C999'],
            '力量名': ['謎のスキル'],  # どのキーワードにもマッチしない
        })

        hierarchy = SkillDomainHierarchy(competence_master)

        # Noneまたはフォールバック分類
        domain = hierarchy.get_domain('C999')
        assert domain is None or domain == 'プログラミング'  # フォールバックの場合


class TestDomainRetrieval:
    """ドメイン取得機能のテスト"""

    def test_get_competences_by_domain(self, sample_competence_master):
        """特定ドメインの力量取得テスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        programming_competences = hierarchy.get_competences_by_domain('プログラミング')

        # プログラミング領域の力量が取得される
        assert len(programming_competences) > 0
        assert 'C001' in programming_competences  # Python基礎
        assert 'C003' in programming_competences  # Web開発

    def test_get_competences_by_level(self, sample_competence_master):
        """特定レベルの力量取得テスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # プログラミング Level 1
        level_1_competences = hierarchy.get_competences_by_level('プログラミング', 1)
        assert len(level_1_competences) > 0
        assert 'C001' in level_1_competences  # Python基礎

        # プログラミング Level 2
        level_2_competences = hierarchy.get_competences_by_level('プログラミング', 2)
        assert len(level_2_competences) > 0
        assert 'C003' in level_2_competences  # Web開発

    def test_get_next_level_competences(self, sample_competence_master):
        """次のレベルの力量取得テスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # Python基礎（Level 1）の次のレベル（Level 2）
        next_level = hierarchy.get_next_level_competences('C001')

        assert len(next_level) > 0
        # Web開発（Level 2）が含まれる
        assert 'C003' in next_level

    def test_get_next_level_for_max_level(self, sample_competence_master):
        """最高レベルの力量の次のレベル取得テスト（空リストが返る）"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # システム設計（Level 3）の次のレベルは存在しない
        next_level = hierarchy.get_next_level_competences('C005')

        assert len(next_level) == 0


class TestDomainProgression:
    """ドメイン進行経路のテスト"""

    def test_get_domain_progression(self, sample_competence_master):
        """ドメインの進行経路取得テスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        progression = hierarchy.get_domain_progression('プログラミング')

        # 3つのレベルが存在
        assert 1 in progression
        assert 2 in progression
        assert 3 in progression

        # 各レベルに力量が含まれる
        assert len(progression[1]) > 0  # Level 1
        assert len(progression[2]) > 0  # Level 2
        assert len(progression[3]) > 0  # Level 3


class TestDomainStatistics:
    """統計機能のテスト"""

    def test_get_domain_statistics(self, sample_competence_master):
        """ドメイン統計取得テスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        stats_df = hierarchy.get_domain_statistics()

        # DataFrameが返される
        assert isinstance(stats_df, pd.DataFrame)

        # 必須カラムが存在
        assert 'Domain' in stats_df.columns
        assert 'Level_1' in stats_df.columns
        assert 'Level_2' in stats_df.columns
        assert 'Level_3' in stats_df.columns
        assert 'Total' in stats_df.columns

        # 各ドメインが行として存在
        domains = stats_df['Domain'].tolist()
        assert 'プログラミング' in domains
        assert 'データベース' in domains

    def test_statistics_values_valid(self, sample_competence_master):
        """統計値が妥当かテスト"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        stats_df = hierarchy.get_domain_statistics()

        # すべての値が非負整数
        for col in ['Level_1', 'Level_2', 'Level_3', 'Total']:
            assert (stats_df[col] >= 0).all()

        # Total = Level_1 + Level_2 + Level_3
        calculated_total = stats_df['Level_1'] + stats_df['Level_2'] + stats_df['Level_3']
        assert (stats_df['Total'] == calculated_total).all()


class TestCustomHierarchy:
    """カスタム階層定義のテスト"""

    def test_custom_hierarchy(self):
        """カスタム階層定義が使えるかテスト"""
        custom_hierarchy = {
            'カスタム領域': {
                'level_1': {
                    'keywords': ['基礎A', '基礎B'],
                    'description': 'カスタム初級',
                },
                'level_2': {
                    'keywords': ['応用A', '応用B'],
                    'description': 'カスタム中級',
                },
                'level_3': {
                    'keywords': ['上級A', '上級B'],
                    'description': 'カスタム上級',
                },
            }
        }

        competence_master = pd.DataFrame({
            '力量コード': ['C001', 'C002'],
            '力量名': ['基礎A', '応用A'],
        })

        hierarchy = SkillDomainHierarchy(
            competence_master,
            custom_hierarchy=custom_hierarchy
        )

        # カスタム階層が使われている
        assert 'カスタム領域' in hierarchy.hierarchy

        # 基礎A → カスタム領域 Level 1
        assert hierarchy.get_domain('C001') == 'カスタム領域'
        assert hierarchy.get_level('C001') == 1

        # 応用A → カスタム領域 Level 2
        assert hierarchy.get_domain('C002') == 'カスタム領域'
        assert hierarchy.get_level('C002') == 2


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_competence_master(self):
        """空の力量マスタ"""
        empty_df = pd.DataFrame({
            '力量コード': [],
            '力量名': [],
        })

        hierarchy = SkillDomainHierarchy(empty_df)

        # エラーなく初期化される
        assert hierarchy is not None
        assert len(hierarchy.competence_classification) == 0

    def test_missing_competence_code(self, sample_competence_master):
        """存在しない力量コードの取得"""
        hierarchy = SkillDomainHierarchy(sample_competence_master)

        # Noneが返される
        assert hierarchy.get_domain('C9999') is None
        assert hierarchy.get_level('C9999') is None
