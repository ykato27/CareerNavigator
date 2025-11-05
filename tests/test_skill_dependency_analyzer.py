"""
スキル依存関係分析のテスト
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from skillnote_recommendation.core.skill_dependency_analyzer import (
    SkillDependencyAnalyzer,
    SkillTransition,
    LearningPath
)


@pytest.fixture
def sample_competence_master():
    """テスト用の力量マスタ"""
    return pd.DataFrame({
        '力量コード': ['c001', 'c002', 'c003', 'c004', 'c005'],
        '力量名': ['Python基礎', 'SQL基礎', 'データ分析', 'Python機械学習', '深層学習'],
        '力量タイプ': ['SKILL', 'SKILL', 'SKILL', 'SKILL', 'SKILL'],
        '力量カテゴリー名': ['プログラミング', 'データベース', 'データサイエンス', 'AI', 'AI']
    })


@pytest.fixture
def sample_member_competence_with_dates():
    """テスト用のメンバー習得力量データ（取得日付き）"""
    base_date = datetime(2023, 1, 1)

    # メンバー1: Python基礎 → データ分析 → Python機械学習
    m1_data = [
        {'メンバーコード': 'm001', '力量コード': 'c001', '正規化レベル': 3, '取得日': base_date.strftime('%Y/%m/%d')},
        {'メンバーコード': 'm001', '力量コード': 'c003', '正規化レベル': 2, '取得日': (base_date + timedelta(days=60)).strftime('%Y/%m/%d')},
        {'メンバーコード': 'm001', '力量コード': 'c004', '正規化レベル': 2, '取得日': (base_date + timedelta(days=150)).strftime('%Y/%m/%d')},
    ]

    # メンバー2: Python基礎 → Python機械学習 → 深層学習
    m2_data = [
        {'メンバーコード': 'm002', '力量コード': 'c001', '正規化レベル': 4, '取得日': (base_date + timedelta(days=10)).strftime('%Y/%m/%d')},
        {'メンバーコード': 'm002', '力量コード': 'c004', '正規化レベル': 3, '取得日': (base_date + timedelta(days=120)).strftime('%Y/%m/%d')},
        {'メンバーコード': 'm002', '力量コード': 'c005', '正規化レベル': 2, '取得日': (base_date + timedelta(days=250)).strftime('%Y/%m/%d')},
    ]

    # メンバー3: Python基礎 → データ分析 → Python機械学習
    m3_data = [
        {'メンバーコード': 'm003', '力量コード': 'c001', '正規化レベル': 3, '取得日': (base_date + timedelta(days=20)).strftime('%Y/%m/%d')},
        {'メンバーコード': 'm003', '力量コード': 'c003', '正規化レベル': 3, '取得日': (base_date + timedelta(days=80)).strftime('%Y/%m/%d')},
        {'メンバーコード': 'm003', '力量コード': 'c004', '正規化レベル': 3, '取得日': (base_date + timedelta(days=180)).strftime('%Y/%m/%d')},
    ]

    # メンバー4: SQL基礎 → データ分析
    m4_data = [
        {'メンバーコード': 'm004', '力量コード': 'c002', '正規化レベル': 4, '取得日': (base_date + timedelta(days=30)).strftime('%Y/%m/%d')},
        {'メンバーコード': 'm004', '力量コード': 'c003', '正規化レベル': 3, '取得日': (base_date + timedelta(days=100)).strftime('%Y/%m/%d')},
    ]

    # メンバー5: Python基礎 → Python機械学習
    m5_data = [
        {'メンバーコード': 'm005', '力量コード': 'c001', '正規化レベル': 5, '取得日': (base_date + timedelta(days=5)).strftime('%Y/%m/%d')},
        {'メンバーコード': 'm005', '力量コード': 'c004', '正規化レベル': 4, '取得日': (base_date + timedelta(days=140)).strftime('%Y/%m/%d')},
    ]

    all_data = m1_data + m2_data + m3_data + m4_data + m5_data
    return pd.DataFrame(all_data)


class TestSkillDependencyAnalyzer:
    """SkillDependencyAnalyzerのテスト"""

    def test_initialization(self, sample_member_competence_with_dates, sample_competence_master):
        """初期化のテスト"""
        analyzer = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master
        )

        assert analyzer is not None
        assert len(analyzer.member_competence) > 0
        assert '取得日_dt' in analyzer.member_competence.columns

    def test_initialization_without_date_column(self, sample_competence_master):
        """取得日カラムがない場合のエラーテスト"""
        member_competence = pd.DataFrame({
            'メンバーコード': ['m001'],
            '力量コード': ['c001'],
            '正規化レベル': [3]
        })

        with pytest.raises(ValueError, match="'取得日'カラムが必要です"):
            SkillDependencyAnalyzer(
                member_competence=member_competence,
                competence_master=sample_competence_master
            )

    def test_extract_temporal_transitions(self, sample_member_competence_with_dates, sample_competence_master):
        """時系列遷移の抽出テスト"""
        analyzer = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master
        )

        transitions_df = analyzer.extract_temporal_transitions()

        assert not transitions_df.empty
        assert 'prerequisite_code' in transitions_df.columns
        assert 'dependent_code' in transitions_df.columns
        assert 'time_gap_days' in transitions_df.columns

        # Python基礎 → Python機械学習の遷移が複数あるはず
        python_to_ml = transitions_df[
            (transitions_df['prerequisite_code'] == 'c001') &
            (transitions_df['dependent_code'] == 'c004')
        ]
        assert len(python_to_ml) >= 2

    def test_calculate_transition_confidence(self, sample_member_competence_with_dates, sample_competence_master):
        """遷移信頼度の計算テスト"""
        analyzer = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master,
            min_transition_count=2
        )

        transitions_df = analyzer.extract_temporal_transitions()
        confidences_df = analyzer.calculate_transition_confidence(transitions_df)

        assert not confidences_df.empty
        assert 'confidence' in confidences_df.columns
        assert 'transition_count' in confidences_df.columns

        # 信頼度が0-1の範囲内であることを確認
        assert (confidences_df['confidence'] >= 0).all()
        assert (confidences_df['confidence'] <= 1).all()

    def test_infer_dependency_direction(self, sample_member_competence_with_dates, sample_competence_master):
        """依存関係の方向推定テスト"""
        analyzer = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master,
            min_transition_count=2,
            confidence_threshold=0.3
        )

        transitions_df = analyzer.extract_temporal_transitions()
        confidences_df = analyzer.calculate_transition_confidence(transitions_df)
        skill_transitions = analyzer.infer_dependency_direction(confidences_df)

        assert len(skill_transitions) > 0

        # Python基礎 → Python機械学習の依存関係があるはず
        python_to_ml = [
            t for t in skill_transitions
            if t.prerequisite_code == 'c001' and t.dependent_code == 'c004'
        ]
        assert len(python_to_ml) > 0

        transition = python_to_ml[0]
        assert isinstance(transition, SkillTransition)
        assert transition.confidence > 0
        assert transition.dependency_strength in ['強', '中', '弱', 'なし']

    def test_find_parallel_learnable_skills(self, sample_competence_master):
        """並列学習可能スキルの特定テスト"""
        # 双方向の遷移が同程度のデータを作成
        base_date = datetime(2023, 1, 1)
        data = []

        # 3人がPython→SQL、3人がSQL→Python
        for i in range(3):
            data.append({
                'メンバーコード': f'm{i+1}',
                '力量コード': 'c001',
                '正規化レベル': 3,
                '取得日': base_date.strftime('%Y/%m/%d')
            })
            data.append({
                'メンバーコード': f'm{i+1}',
                '力量コード': 'c002',
                '正規化レベル': 3,
                '取得日': (base_date + timedelta(days=60)).strftime('%Y/%m/%d')
            })

        for i in range(3, 6):
            data.append({
                'メンバーコード': f'm{i+1}',
                '力量コード': 'c002',
                '正規化レベル': 3,
                '取得日': base_date.strftime('%Y/%m/%d')
            })
            data.append({
                'メンバーコード': f'm{i+1}',
                '力量コード': 'c001',
                '正規化レベル': 3,
                '取得日': (base_date + timedelta(days=60)).strftime('%Y/%m/%d')
            })

        member_competence = pd.DataFrame(data)

        analyzer = SkillDependencyAnalyzer(
            member_competence=member_competence,
            competence_master=sample_competence_master,
            min_transition_count=2
        )

        transitions_df = analyzer.extract_temporal_transitions()
        confidences_df = analyzer.calculate_transition_confidence(transitions_df)
        parallel_skills = analyzer.find_parallel_learnable_skills(confidences_df)

        # Python基礎とSQL基礎が並列学習可能として検出されるはず
        assert len(parallel_skills) > 0

    def test_generate_learning_paths(self, sample_member_competence_with_dates, sample_competence_master):
        """学習パス生成のテスト"""
        analyzer = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master,
            min_transition_count=2
        )

        learning_paths = analyzer.generate_learning_paths()

        assert len(learning_paths) > 0

        # Python機械学習の学習パスを確認
        if 'c004' in learning_paths:
            ml_path = learning_paths['c004']
            assert isinstance(ml_path, LearningPath)
            assert ml_path.competence_name == 'Python機械学習'
            assert ml_path.estimated_difficulty in ['初級', '中級', '上級']

            # Python基礎が前提スキルに含まれているはず
            prerequisite_codes = [p['skill_code'] for p in ml_path.recommended_prerequisites]
            assert 'c001' in prerequisite_codes

    def test_generate_learning_paths_for_specific_skills(self, sample_member_competence_with_dates, sample_competence_master):
        """特定スキルの学習パス生成テスト"""
        analyzer = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master
        )

        target_codes = ['c004', 'c005']  # Python機械学習と深層学習
        learning_paths = analyzer.generate_learning_paths(target_competence_codes=target_codes)

        assert len(learning_paths) <= len(target_codes)
        assert all(code in target_codes for code in learning_paths.keys())

    def test_get_dependency_graph_data(self, sample_member_competence_with_dates, sample_competence_master):
        """依存関係グラフデータ取得のテスト"""
        analyzer = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master,
            min_transition_count=2
        )

        graph_data = analyzer.get_dependency_graph_data()

        assert 'nodes' in graph_data
        assert 'edges' in graph_data
        assert isinstance(graph_data['nodes'], list)
        assert isinstance(graph_data['edges'], list)

        if graph_data['edges']:
            edge = graph_data['edges'][0]
            assert 'source' in edge
            assert 'target' in edge
            assert 'weight' in edge
            assert 'strength' in edge

    def test_empty_transitions(self, sample_competence_master):
        """遷移がない場合のテスト"""
        # 単一メンバー、単一スキルのデータ
        member_competence = pd.DataFrame({
            'メンバーコード': ['m001'],
            '力量コード': ['c001'],
            '正規化レベル': [3],
            '取得日': ['2023/01/01']
        })

        analyzer = SkillDependencyAnalyzer(
            member_competence=member_competence,
            competence_master=sample_competence_master
        )

        transitions_df = analyzer.extract_temporal_transitions()
        assert transitions_df.empty

        learning_paths = analyzer.generate_learning_paths()
        # 遷移がなくても学習パスは生成される（前提スキルなしとして）
        assert isinstance(learning_paths, dict)

    def test_confidence_threshold_filtering(self, sample_member_competence_with_dates, sample_competence_master):
        """信頼度閾値によるフィルタリングのテスト"""
        # 高い閾値
        analyzer_high = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master,
            confidence_threshold=0.8
        )

        # 低い閾値
        analyzer_low = SkillDependencyAnalyzer(
            member_competence=sample_member_competence_with_dates,
            competence_master=sample_competence_master,
            confidence_threshold=0.2
        )

        paths_high = analyzer_high.generate_learning_paths()
        paths_low = analyzer_low.generate_learning_paths()

        # 低い閾値の方が、より多くの依存関係を検出するはず
        total_prereqs_high = sum(len(p.recommended_prerequisites) for p in paths_high.values())
        total_prereqs_low = sum(len(p.recommended_prerequisites) for p in paths_low.values())

        assert total_prereqs_low >= total_prereqs_high


class TestLearningPath:
    """LearningPathのテスト"""

    def test_learning_path_attributes(self):
        """LearningPathの属性テスト"""
        path = LearningPath(
            competence_code='c001',
            competence_name='Python基礎',
            competence_type='SKILL',
            category='プログラミング',
            recommended_prerequisites=[],
            can_learn_in_parallel=[],
            unlocks=[],
            estimated_difficulty='初級',
            estimated_learning_hours=None,
            success_rate=0.85
        )

        assert path.competence_code == 'c001'
        assert path.estimated_difficulty == '初級'
        assert path.success_rate == 0.85


class TestSkillTransition:
    """SkillTransitionのテスト"""

    def test_skill_transition_attributes(self):
        """SkillTransitionの属性テスト"""
        transition = SkillTransition(
            prerequisite_code='c001',
            prerequisite_name='Python基礎',
            dependent_code='c004',
            dependent_name='Python機械学習',
            transition_count=10,
            median_time_gap_days=90.0,
            confidence=0.75,
            dependency_strength='強',
            reverse_transition_count=2,
            evidence='10人がPython基礎→Python機械学習の順序で学習'
        )

        assert transition.prerequisite_code == 'c001'
        assert transition.dependent_code == 'c004'
        assert transition.confidence == 0.75
        assert transition.dependency_strength == '強'
