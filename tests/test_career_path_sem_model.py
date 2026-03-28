"""
CareerPathSEMModelのテスト

キャリアパス因果構造モデルの学習・推薦・評価のテスト
"""

import pytest
import pandas as pd
import numpy as np

from skillnote_recommendation.ml.career_path_sem_model import CareerPathSEMModel
from skillnote_recommendation.ml.career_path_hierarchy import CareerPathHierarchy


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータ"""
    # メンバーマスタ
    member_master = pd.DataFrame({
        'メンバーコード': ['M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007'],
        '氏名': ['山田太郎', '佐藤花子', '鈴木一郎', '田中美咲', '高橋健太', '伊藤次郎', '渡辺美穂'],
        '役職': ['一般社員', '一般社員', '主任', '主任', '主任', '課長', '課長'],
    })

    # 力量マスタ
    competence_master = pd.DataFrame({
        '力量コード': [
            'C001', 'C002', 'C003', 'C004', 'C005',  # 一般社員・入門期
            'C006', 'C007', 'C008', 'C009', 'C010',  # 一般社員・成長期
            'C011', 'C012', 'C013',                  # 主任・入門期
            'C014', 'C015', 'C016',                  # 主任・成長期
            'C017', 'C018', 'C019',                  # 課長・入門期
            'C020', 'C021', 'C022',                  # 課長・成長期
        ],
        '力量名': [
            'ビジネスマナー基礎', '報連相', 'メール作成', '資料作成', 'スケジュール管理',
            '業務遂行', 'プロジェクト参加', '品質管理', '顧客対応', '改善提案',
            'リーダーシップ基礎', 'チーム運営', '進捗管理',
            'プロジェクト管理', 'リスク管理', '目標設定',
            'マネジメント基礎', '評価面談', '部下育成',
            '組織戦略', '予算管理', '経営計画',
        ],
    })

    # メンバー力量データ
    member_competence_data = []

    # M001: 一般社員・入門期を一部習得
    for comp in ['C001', 'C002', 'C003']:
        member_competence_data.append({
            'メンバーコード': 'M001',
            '力量コード': comp,
            '正規化レベル': 0.8,
        })

    # M002: 一般社員・入門期+成長期を習得
    for comp in ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007']:
        member_competence_data.append({
            'メンバーコード': 'M002',
            '力量コード': comp,
            '正規化レベル': 0.7,
        })

    # M003: 主任・入門期を習得
    for comp in ['C011', 'C012', 'C013']:
        member_competence_data.append({
            'メンバーコード': 'M003',
            '力量コード': comp,
            '正規化レベル': 0.9,
        })

    # M004: 主任・入門期+成長期を習得
    for comp in ['C011', 'C012', 'C013', 'C014', 'C015']:
        member_competence_data.append({
            'メンバーコード': 'M004',
            '力量コード': comp,
            '正規化レベル': 0.75,
        })

    # M005: 主任・全習得
    for comp in ['C011', 'C012', 'C013', 'C014', 'C015', 'C016']:
        member_competence_data.append({
            'メンバーコード': 'M005',
            '力量コード': comp,
            '正規化レベル': 0.85,
        })

    # M006: 課長・入門期を習得
    for comp in ['C017', 'C018', 'C019']:
        member_competence_data.append({
            'メンバーコード': 'M006',
            '力量コード': comp,
            '正規化レベル': 0.7,
        })

    # M007: 課長・全習得
    for comp in ['C017', 'C018', 'C019', 'C020', 'C021', 'C022']:
        member_competence_data.append({
            'メンバーコード': 'M007',
            '力量コード': comp,
            '正規化レベル': 0.8,
        })

    member_competence = pd.DataFrame(member_competence_data)

    return member_master, member_competence, competence_master


class TestCareerPathSEMModelInit:
    """初期化のテスト"""

    def test_initialization(self, sample_data):
        """初期化が正常に行われるかテスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )

        assert model is not None
        assert model.career_path_hierarchy is not None
        assert not model.is_fitted

    def test_initialization_with_custom_hierarchy(self, sample_data):
        """カスタム階層を使った初期化テスト"""
        member_master, member_competence, competence_master = sample_data

        # 先に階層を作成
        hierarchy = CareerPathHierarchy(
            member_master, member_competence, competence_master
        )

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master,
            career_path_hierarchy=hierarchy
        )

        assert model is not None
        assert model.career_path_hierarchy is hierarchy


class TestModelFitting:
    """モデル学習のテスト"""

    def test_fit_basic(self, sample_data):
        """基本的な学習テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )

        # 学習実行
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # 学習完了フラグが立つ
        assert model.is_fitted

        # SEMモデルが構築されている
        assert len(model.sem_models) > 0

    def test_fit_with_specific_roles(self, sample_data):
        """特定の役職のみ学習するテスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )

        # 主任のみ学習
        model.fit(roles=['主任'], min_members_per_role=2, min_skills_per_stage=2)

        assert model.is_fitted

        # 主任のSEMモデルが存在
        if '主任' in model.sem_models:
            assert model.sem_models['主任'] is not None

    def test_fit_insufficient_data(self, sample_data):
        """データ不足時の学習テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )

        # 厳しい条件で学習
        model.fit(min_members_per_role=10, min_skills_per_stage=10)

        # エラーなく完了（ただしモデルは構築されない可能性）
        assert model.is_fitted


class TestMemberPosition:
    """メンバー位置推定のテスト"""

    def test_get_member_position(self, sample_data):
        """メンバー位置取得テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003の位置を取得
        role, stage, progress = model.get_member_position('M003')

        assert role == '主任'
        assert isinstance(stage, int)
        assert 0.0 <= progress <= 1.0

    def test_get_position_nonexistent_member(self, sample_data):
        """存在しないメンバーの位置取得テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # 存在しないメンバー
        role, stage, progress = model.get_member_position('M999')

        assert role is None
        assert stage == 0
        assert progress == 0.0


class TestPathAlignmentScore:
    """Path Alignment Scoreのテスト"""

    def test_calculate_path_alignment_score(self, sample_data):
        """Path Alignment Score計算テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003（主任・入門期）に対して、成長期のスキルを評価
        score = model.calculate_path_alignment_score('M003', 'C014')

        # スコアが0.0～1.0の範囲
        assert 0.0 <= score <= 1.0

    def test_score_for_current_stage_skill(self, sample_data):
        """現在のステージのスキルのスコアテスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003が現在いるステージのスキル（未習得）を推薦
        # M003はC011,C012,C013を習得済み
        # 同じステージの他のスキルは存在しないため、次のステージのスキルC014を評価

        score = model.calculate_path_alignment_score('M003', 'C014')

        # 次のステージのスキル → 0.8前後のスコア
        assert score > 0.5

    def test_score_for_off_path_skill(self, sample_data):
        """パス上にないスキルのスコアテスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003（主任）に対して、一般社員のスキルを評価
        score = model.calculate_path_alignment_score('M003', 'C001')

        # パス上にないスキル → 0.0
        assert score == 0.0


class TestPathExplanation:
    """推薦理由生成のテスト"""

    def test_generate_path_explanation(self, sample_data):
        """推薦理由生成テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003に対してC014の推薦理由を生成
        explanation = model.generate_path_explanation('M003', 'C014')

        # 説明文が生成される
        assert isinstance(explanation, str)
        assert len(explanation) > 0

        # キーワードが含まれる
        assert 'キャリアパス' in explanation or '推薦' in explanation

    def test_explanation_includes_path_score(self, sample_data):
        """推薦理由にPath Alignment Scoreが含まれるかテスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        explanation = model.generate_path_explanation('M003', 'C014')

        # Path Alignment Scoreの記載がある
        assert 'Path Alignment Score' in explanation

    def test_explanation_for_off_path_skill(self, sample_data):
        """パス上にないスキルの推薦理由テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003（主任）に対して、一般社員のスキルC001の推薦理由
        explanation = model.generate_path_explanation('M003', 'C001')

        # パス上にないことが明記される
        assert 'パス上にありません' in explanation or 'キャリアパス' in explanation


class TestRecommendNextSteps:
    """次のステップ推薦のテスト"""

    def test_recommend_next_steps(self, sample_data):
        """次のステップ推薦テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003の次のステップを推薦
        recommendations = model.recommend_next_steps('M003', top_n=5)

        # 推薦結果が返される
        assert isinstance(recommendations, list)

        # 推薦結果の構造
        if len(recommendations) > 0:
            for rec in recommendations:
                assert 'competence_code' in rec
                assert 'competence_name' in rec

    def test_recommendations_include_path_coefficient(self, sample_data):
        """推薦結果にパス係数が含まれるかテスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        recommendations = model.recommend_next_steps('M003', top_n=5)

        # パス係数情報が含まれる
        if len(recommendations) > 0:
            # すべての推薦にpath_coefficientキーが存在
            assert all('path_coefficient' in rec for rec in recommendations)


class TestCareerProgressionSummary:
    """キャリア進捗サマリーのテスト"""

    def test_get_career_progression_summary(self, sample_data):
        """キャリア進捗サマリー取得テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # M003のサマリーを取得
        summary = model.get_career_progression_summary('M003')

        # サマリーの構造
        assert isinstance(summary, dict)

        # 必須キーが存在
        if len(summary) > 0:
            assert 'role' in summary
            assert 'current_stage' in summary
            assert 'current_stage_name' in summary
            assert 'progress' in summary


class TestRolePathSummary:
    """役職パスサマリーのテスト"""

    def test_get_role_path_summary(self, sample_data):
        """役職パスサマリー取得テスト"""
        member_master, member_competence, competence_master = sample_data

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )
        model.fit(min_members_per_role=2, min_skills_per_stage=2)

        # 主任のパスサマリーを取得
        summary_df = model.get_role_path_summary('主任')

        # DataFrameが返される
        assert isinstance(summary_df, pd.DataFrame)

        # カラムが存在
        if len(summary_df) > 0:
            assert 'Stage' in summary_df.columns
            assert 'Stage_Name' in summary_df.columns
            assert 'Skills' in summary_df.columns
            assert 'Path_Coefficient' in summary_df.columns


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_member_competence(self):
        """空のメンバー力量データ"""
        member_master = pd.DataFrame({
            'メンバーコード': ['M001'],
            '氏名': ['テスト'],
            '役職': ['一般社員'],
        })

        competence_master = pd.DataFrame({
            '力量コード': ['C001'],
            '力量名': ['テストスキル'],
        })

        member_competence = pd.DataFrame({
            'メンバーコード': [],
            '力量コード': [],
            '正規化レベル': [],
        })

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )

        # エラーなく初期化される
        assert model is not None

        # 学習もエラーなく完了（ただしモデルは構築されない）
        model.fit(min_members_per_role=1, min_skills_per_stage=1)
        assert model.is_fitted

    def test_single_member_role(self):
        """1名のみの役職"""
        member_master = pd.DataFrame({
            'メンバーコード': ['M001'],
            '氏名': ['テスト'],
            '役職': ['一般社員'],
        })

        competence_master = pd.DataFrame({
            '力量コード': ['C001', 'C002'],
            '力量名': ['スキルA', 'スキルB'],
        })

        member_competence = pd.DataFrame({
            'メンバーコード': ['M001', 'M001'],
            '力量コード': ['C001', 'C002'],
            '正規化レベル': [0.8, 0.7],
        })

        model = CareerPathSEMModel(
            member_master, member_competence, competence_master
        )

        # 最小人数を1に設定
        model.fit(min_members_per_role=1, min_skills_per_stage=1)

        assert model.is_fitted
