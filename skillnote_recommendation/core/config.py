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
        # 基本パラメータ
        'n_components': 20,  # 潜在因子の数（10-30推奨）
        'max_iter': 1000,  # 最大イテレーション数（500-2000推奨）
        'random_state': 42,  # 再現性のための乱数シード

        # 収束パラメータ
        'tol': 1e-5,  # 収束判定の閾値（1e-4 → 1e-5で精度向上）

        # 初期化戦略
        'init': 'nndsvda',  # 'nndsvda' (sparse data向け), 'nndsvd', 'random'

        # 正則化パラメータ（過学習防止）
        'alpha_W': 0.01,  # メンバー因子行列のL1/L2正則化（0.0-0.1推奨）
        'alpha_H': 0.01,  # 力量因子行列のL1/L2正則化（0.0-0.1推奨）
        'l1_ratio': 0.5,  # L1正則化の割合（0.0=L2のみ, 1.0=L1のみ）

        # ソルバー
        'solver': 'cd',  # 'cd' (coordinate descent) or 'mu' (multiplicative update)
    }

    # データ前処理パラメータ
    DATA_PREPROCESSING_PARAMS = {
        'min_competences_per_member': 3,  # メンバーが保有すべき最小力量数（外れ値除去用）
        'min_members_per_competence': 3,  # 力量を保有すべき最小メンバー数（外れ値除去用）
        'normalization_method': 'minmax',  # 'minmax', 'standard', 'l2', None
        'enable_preprocessing': True,  # 前処理を有効にするか
    }

    # Optunaハイパーパラメータチューニングパラメータ
    OPTUNA_PARAMS = {
        'n_trials': 50,  # 試行回数
        'timeout': 600,  # タイムアウト（秒）
        'n_jobs': 1,  # 並列実行数（1=逐次実行）
        'show_progress_bar': True,  # プログレスバーを表示
        'search_space': {
            'n_components': (10, 40),  # 探索範囲（最小, 最大）
            'alpha_W': (0.0, 0.2),  # 対数スケールで探索
            'alpha_H': (0.0, 0.2),
            'l1_ratio': (0.0, 1.0),
            'max_iter': (500, 2000),
        },
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
