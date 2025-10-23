"""
設定ファイル

システム全体で使用する設定値を管理
"""

import os


class Config:
    """システム設定クラス"""
    
    # ディレクトリ設定
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    
    # 入力ファイル名
    INPUT_FILES = {
        'members': 'member_skillnote.csv',
        'acquired': 'acquiredCompetenceLevel.csv',
        'skills': 'skill_skillnote.csv',
        'education': 'education_skillnote.csv',
        'license': 'license_skillnote.csv',
        'categories': 'competence_category_skillnote.csv'
    }
    
    # 出力ファイル名
    OUTPUT_FILES = {
        'members_clean': 'members_clean.csv',
        'competence_master': 'competence_master.csv',
        'member_competence': 'member_competence.csv',
        'skill_matrix': 'skill_matrix.csv',
        'competence_similarity': 'competence_similarity.csv'
    }
    
    # 推薦システムパラメータ
    RECOMMENDATION_PARAMS = {
        'category_importance_weight': 0.4,
        'acquisition_ease_weight': 0.3,
        'popularity_weight': 0.3,
        'similarity_threshold': 0.3,  # 類似度の閾値
        'similarity_sample_size': 100  # 類似度計算のサンプル数
    }
    
    # エンコーディング
    FILE_ENCODING = 'utf-8'
    OUTPUT_ENCODING = 'utf-8-sig'
    
    @classmethod
    def get_input_path(cls, file_key):
        """入力ファイルのパスを取得"""
        return os.path.join(cls.DATA_DIR, cls.INPUT_FILES[file_key])
    
    @classmethod
    def get_output_path(cls, file_key):
        """出力ファイルのパスを取得"""
        return os.path.join(cls.OUTPUT_DIR, cls.OUTPUT_FILES[file_key])
    
    @classmethod
    def ensure_directories(cls):
        """必要なディレクトリを作成"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
