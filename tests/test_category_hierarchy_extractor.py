"""
カテゴリ階層抽出のテスト
"""

import pytest
import pandas as pd
import os
from pathlib import Path

from skillnote_recommendation.ml.category_hierarchy_extractor import (
    CategoryHierarchy,
    CategoryHierarchyExtractor
)


@pytest.fixture
def data_dir():
    """データディレクトリのパスを取得"""
    # プロジェクトルートからの相対パス
    project_root = Path(__file__).parent.parent
    return project_root / 'data'


@pytest.fixture
def extractor(data_dir):
    """CategoryHierarchyExtractorのインスタンスを作成"""
    category_csv = data_dir / 'categories' / 'competence_category_skillnote.csv'
    skill_csv = data_dir / 'skills' / 'skill_skillnote.csv'
    
    return CategoryHierarchyExtractor(
        category_csv_path=str(category_csv),
        skill_csv_path=str(skill_csv)
    )


class TestCategoryHierarchy:
    """CategoryHierarchyクラスのテスト"""
    
    def test_get_level(self):
        """レベル取得のテスト"""
        hierarchy = CategoryHierarchy()
        hierarchy.level1_categories = ['CTG100000000']
        hierarchy.level2_categories = ['CTG101000000']
        hierarchy.level3_categories = ['CTG101010000']
        
        assert hierarchy.get_level('CTG100000000') == 1
        assert hierarchy.get_level('CTG101000000') == 2
        assert hierarchy.get_level('CTG101010000') == 3
        assert hierarchy.get_level('UNKNOWN') == 0
    
    def test_get_ancestors(self):
        """祖先取得のテスト"""
        hierarchy = CategoryHierarchy()
        hierarchy.parent_mapping = {
            'CTG101010000': 'CTG101000000',
            'CTG101000000': 'CTG100000000'
        }
        
        ancestors = hierarchy.get_ancestors('CTG101010000')
        assert ancestors == ['CTG101000000', 'CTG100000000']
        
        ancestors = hierarchy.get_ancestors('CTG101000000')
        assert ancestors == ['CTG100000000']
        
        ancestors = hierarchy.get_ancestors('CTG100000000')
        assert ancestors == []
    
    def test_get_l1_category(self):
        """L1カテゴリ取得のテスト"""
        hierarchy = CategoryHierarchy()
        hierarchy.level1_categories = ['CTG100000000']
        hierarchy.level2_categories = ['CTG101000000']
        hierarchy.level3_categories = ['CTG101010000']
        hierarchy.parent_mapping = {
            'CTG101010000': 'CTG101000000',
            'CTG101000000': 'CTG100000000'
        }
        
        # L1自身
        assert hierarchy.get_l1_category('CTG100000000') == 'CTG100000000'
        
        # L2からL1
        assert hierarchy.get_l1_category('CTG101000000') == 'CTG100000000'
        
        # L3からL1
        assert hierarchy.get_l1_category('CTG101010000') == 'CTG100000000'


class TestCategoryHierarchyExtractor:
    """CategoryHierarchyExtractorクラスのテスト"""
    
    def test_extract_hierarchy(self, extractor):
        """階層抽出のテスト"""
        hierarchy = extractor.extract_hierarchy()
        
        # 階層が抽出されていることを確認
        assert isinstance(hierarchy, CategoryHierarchy)
        
        # L1カテゴリが存在することを確認（10-20個程度）
        assert 5 <= len(hierarchy.level1_categories) <= 30
        print(f"L1カテゴリ数: {len(hierarchy.level1_categories)}")
        
        # L2カテゴリが存在することを確認（30-50個程度）
        assert 10 <= len(hierarchy.level2_categories) <= 100
        print(f"L2カテゴリ数: {len(hierarchy.level2_categories)}")
        
        # L3カテゴリが存在することを確認（100個以上）
        assert len(hierarchy.level3_categories) >= 50
        print(f"L3カテゴリ数: {len(hierarchy.level3_categories)}")
        
        # カテゴリ名が設定されていることを確認
        assert len(hierarchy.category_names) > 0
        
        # スキルマッピングが存在することを確認
        assert len(hierarchy.skill_to_category) > 0
        print(f"スキル数: {len(hierarchy.skill_to_category)}")
    
    def test_parent_child_relationships(self, extractor):
        """親子関係のテスト"""
        hierarchy = extractor.extract_hierarchy()
        
        # 親子関係が構築されていることを確認
        assert len(hierarchy.parent_mapping) > 0
        assert len(hierarchy.children_mapping) > 0
        
        # L2カテゴリの親がL1であることを確認
        for l2_code in hierarchy.level2_categories[:5]:  # 最初の5個をチェック
            if l2_code in hierarchy.parent_mapping:
                parent = hierarchy.parent_mapping[l2_code]
                assert parent in hierarchy.level1_categories, \
                    f"L2カテゴリ {l2_code} の親 {parent} がL1カテゴリではありません"
        
        # L3カテゴリの親がL2であることを確認
        for l3_code in hierarchy.level3_categories[:5]:  # 最初の5個をチェック
            if l3_code in hierarchy.parent_mapping:
                parent = hierarchy.parent_mapping[l3_code]
                # 親はL2またはL1のはず
                assert (parent in hierarchy.level2_categories or 
                       parent in hierarchy.level1_categories), \
                    f"L3カテゴリ {l3_code} の親 {parent} が適切なレベルではありません"
    
    def test_skill_category_mapping(self, extractor):
        """スキル-カテゴリマッピングのテスト"""
        hierarchy = extractor.extract_hierarchy()
        
        # スキルが259個程度あることを確認
        assert 200 <= len(hierarchy.skill_to_category) <= 300
        
        # すべてのスキルがカテゴリにマッピングされていることを確認
        for skill_code, category_code in list(hierarchy.skill_to_category.items())[:10]:
            # カテゴリコードが存在することを確認
            assert category_code in hierarchy.category_names, \
                f"スキル {skill_code} のカテゴリ {category_code} が存在しません"
        
        # 逆マッピングも正しいことを確認
        for category_code, skill_list in list(hierarchy.category_to_skills.items())[:10]:
            for skill_code in skill_list:
                assert hierarchy.skill_to_category[skill_code] == category_code, \
                    f"逆マッピングが一致しません: {skill_code}"
    
    def test_get_skills_by_category(self, extractor):
        """カテゴリ別スキル取得のテスト"""
        hierarchy = extractor.extract_hierarchy()
        
        # L3カテゴリのスキルを取得
        if hierarchy.level3_categories:
            l3_code = hierarchy.level3_categories[0]
            skills = extractor.get_skills_by_category(l3_code, include_descendants=False)
            
            # スキルが取得できることを確認
            if skills:  # スキルがある場合
                assert len(skills) > 0
                print(f"L3カテゴリ {l3_code} のスキル数: {len(skills)}")
        
        # L1カテゴリのスキルを取得（子孫含む）
        if hierarchy.level1_categories:
            l1_code = hierarchy.level1_categories[0]
            skills_with_descendants = extractor.get_skills_by_category(
                l1_code, 
                include_descendants=True
            )
            skills_without_descendants = extractor.get_skills_by_category(
                l1_code, 
                include_descendants=False
            )
            
            # 子孫を含む方が多いはず（または同じ）
            assert len(skills_with_descendants) >= len(skills_without_descendants)
            print(f"L1カテゴリ {l1_code} のスキル数（子孫含む）: {len(skills_with_descendants)}")
    
    def test_category_names(self, extractor):
        """カテゴリ名のテスト"""
        hierarchy = extractor.extract_hierarchy()
        
        # すべてのカテゴリに名前が設定されていることを確認
        all_categories = (
            hierarchy.level1_categories + 
            hierarchy.level2_categories + 
            hierarchy.level3_categories
        )
        
        for category_code in all_categories[:10]:  # 最初の10個をチェック
            assert category_code in hierarchy.category_names, \
                f"カテゴリ {category_code} に名前が設定されていません"
            
            name = hierarchy.category_names[category_code]
            assert isinstance(name, str)
            assert len(name) > 0
            
            # L1は階層が1つ、L2は2つ、L3は3つ以上の階層を持つはず
            level = hierarchy.get_level(category_code)
            hierarchy_count = name.count(' > ') + 1
            
            if level == 1:
                assert hierarchy_count == 1, \
                    f"L1カテゴリ {category_code} の階層数が不正: {name}"
            elif level == 2:
                assert hierarchy_count == 2, \
                    f"L2カテゴリ {category_code} の階層数が不正: {name}"
            elif level == 3:
                assert hierarchy_count >= 3, \
                    f"L3カテゴリ {category_code} の階層数が不正: {name}"


def test_integration_with_real_data(extractor):
    """実データでの統合テスト"""
    hierarchy = extractor.extract_hierarchy()
    
    # 統計情報を出力
    print("\n=== カテゴリ階層統計 ===")
    print(f"L1カテゴリ数: {len(hierarchy.level1_categories)}")
    print(f"L2カテゴリ数: {len(hierarchy.level2_categories)}")
    print(f"L3カテゴリ数: {len(hierarchy.level3_categories)}")
    print(f"総カテゴリ数: {len(hierarchy.category_names)}")
    print(f"スキル数: {len(hierarchy.skill_to_category)}")
    print(f"親子関係数: {len(hierarchy.parent_mapping)}")
    
    # L1カテゴリの例を表示
    print("\n=== L1カテゴリの例 ===")
    for l1_code in hierarchy.level1_categories[:5]:
        name = hierarchy.category_names[l1_code]
        children = hierarchy.children_mapping.get(l1_code, [])
        skills = extractor.get_skills_by_category(l1_code, include_descendants=True)
        print(f"{l1_code}: {name}")
        print(f"  子カテゴリ数: {len(children)}")
        print(f"  スキル数（子孫含む）: {len(skills)}")
    
    # 検証: 期待される構造
    assert 5 <= len(hierarchy.level1_categories) <= 30
    assert 10 <= len(hierarchy.level2_categories) <= 100
    assert len(hierarchy.level3_categories) >= 50
    assert 200 <= len(hierarchy.skill_to_category) <= 300
