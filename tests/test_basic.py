"""
基本的なテスト

pytest実行例:
  uv run pytest tests/
"""

import pytest
from skillnote_recommendation.core.config import Config
from skillnote_recommendation.core.models import Member, Competence, Recommendation


def test_config():
    """設定クラスのテスト"""
    assert Config.DATA_DIR.endswith('data')
    assert Config.OUTPUT_DIR.endswith('output')
    assert 'members' in Config.INPUT_DIRS
    assert 'members_clean' in Config.OUTPUT_FILES


def test_member_model():
    """会員モデルのテスト"""
    member = Member(
        member_code='m001',
        name='テスト太郎',
        role='エンジニア',
        grade='3等級'
    )
    
    assert member.member_code == 'm001'
    assert member.name == 'テスト太郎'
    assert member.role == 'エンジニア'
    assert member.grade == '3等級'


def test_competence_model():
    """力量モデルのテスト"""
    competence = Competence(
        competence_code='s001',
        name='Python',
        competence_type='SKILL',
        category='プログラミング',
        description='Pythonプログラミング'
    )
    
    assert competence.competence_code == 's001'
    assert competence.name == 'Python'
    assert competence.competence_type == 'SKILL'


def test_recommendation_model():
    """推薦モデルのテスト"""
    recommendation = Recommendation(
        competence_code='s001',
        competence_name='Python',
        competence_type='SKILL',
        category='プログラミング',
        priority_score=7.5,
        category_importance=8.0,
        acquisition_ease=7.0,
        popularity=6.0,
        reason='推薦理由です'
    )
    
    assert recommendation.competence_code == 's001'
    assert recommendation.priority_score == 7.5
    
    # 辞書変換のテスト
    rec_dict = recommendation.to_dict()
    assert '力量コード' in rec_dict
    assert rec_dict['力量名'] == 'Python'


def test_config_paths():
    """パス取得のテスト"""
    input_dir = Config.get_input_dir('members')
    output_path = Config.get_output_path('members_clean')

    assert 'members' in input_dir
    assert 'members_clean.csv' in output_path


def test_recommendation_params():
    """推薦パラメータのテスト"""
    params = Config.RECOMMENDATION_PARAMS
    
    assert 'category_importance_weight' in params
    assert 'acquisition_ease_weight' in params
    assert 'popularity_weight' in params
    
    # 重みの合計が1.0になることを確認
    total_weight = (
        params['category_importance_weight'] +
        params['acquisition_ease_weight'] +
        params['popularity_weight']
    )
    assert abs(total_weight - 1.0) < 0.01  # 浮動小数点の誤差を考慮
