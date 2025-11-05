"""
設定ファイル

システム全体で使用する設定値を管理
"""

import os


class Config:
    """システム設定クラス"""
    
    # プロジェクトルートディレクトリ
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ディレクトリ設定
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

    # 入力ディレクトリ名（各ディレクトリ内の全CSVを読み込み）
    INPUT_DIRS = {
        'members': 'members',
        'acquired': 'acquired',
        'skills': 'skills',
        'education': 'education',
        'license': 'license',
        'categories': 'categories'
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

    # Knowledge Graph パラメータ
    GRAPH_PARAMS = {
        'member_similarity_threshold': 0.3,  # メンバー類似度の閾値
        'member_similarity_top_k': 5,  # 各メンバーに対する類似メンバー数
    }

    # Matrix Factorization パラメータ
    MF_PARAMS = {
        'n_components': 10,  # 潜在因子の数
        'max_iter': 200,  # 最大イテレーション数
        'random_state': 42,  # 再現性のための乱数シード
    }

    # データ検証パラメータ
    VALIDATION_PARAMS = {
        'min_competences_per_member': 1,  # メンバーが持つべき最小力量数
        'max_name_length': 100,  # 名前の最大長
        'invalid_name_patterns': ['削除', 'テスト', 'test'],  # 無効な名前パターン
    }

    # 可視化パラメータ
    VISUALIZATION_PARAMS = {
        'heatmap_height': 500,  # ヒートマップの高さ（ピクセル）
        'scatter_plot_height': 500,  # 散布図の高さ（ピクセル）
        'max_members_to_show': 10,  # 潜在因子分析で表示する最大メンバー数
        'max_competences_to_show': 10,  # 潜在因子分析で表示する最大力量数
        'color_target_member': '#FF4B4B',  # 対象メンバーの色（赤）
        'color_reference_person': '#4B8BFF',  # 参考人物の色（青）
        'color_other_member': '#CCCCCC',  # その他のメンバーの色（灰色）
    }

    # ログ設定
    LOGGING_PARAMS = {
        'level': 'INFO',  # ログレベル (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
    }

    # エンコーディング
    FILE_ENCODING = 'utf-8'
    OUTPUT_ENCODING = 'utf-8-sig'
    
    @classmethod
    def get_input_dir(cls, dir_key):
        """入力ディレクトリのパスを取得"""
        return os.path.join(cls.DATA_DIR, cls.INPUT_DIRS[dir_key])

    @classmethod
    def get_output_path(cls, file_key):
        """出力ファイルのパスを取得"""
        return os.path.join(cls.OUTPUT_DIR, cls.OUTPUT_FILES[file_key])
    
    @classmethod
    def ensure_directories(cls):
        """必要なディレクトリを作成"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
