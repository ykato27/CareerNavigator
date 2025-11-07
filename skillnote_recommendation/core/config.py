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

    # キャリアパターン別推薦パラメータ
    CAREER_PATTERN_PARAMS = {
        # 類似度の閾値（3段階分類）
        'similar_career_threshold': 0.7,      # この値以上：類似キャリア
        'different_career1_threshold': 0.4,   # この値以上：異なるキャリア1
        # 0.4未満：異なるキャリア2

        # 各パターンの推薦件数
        'similar_career_top_k': 5,            # 類似キャリアからの推薦件数
        'different_career1_top_k': 5,         # 異なるキャリア1からの推薦件数
        'different_career2_top_k': 5,         # 異なるキャリア2からの推薦件数

        # 各パターンの参考人物数
        'similar_career_ref_persons': 5,      # 類似キャリアの参考人物数（最大）
        'different_career1_ref_persons': 5,   # 異なるキャリア1の参考人物数（最大）
        'different_career2_ref_persons': 5,   # 異なるキャリア2の参考人物数（最大）

        # 参考人物の最小数（この数未満の場合は表示しない）
        'min_ref_persons': 1,

        # 参考人物選定基準
        'ref_person_selection': 'top_similar',  # 'top_similar': 類似度上位, 'random': ランダム
    }

    # Knowledge Graph パラメータ
    GRAPH_PARAMS = {
        'member_similarity_threshold': 0.15,  # メンバー類似度の閾値（0.3 → 0.15に緩和）
        'member_similarity_top_k': 10,  # 各メンバーに対する類似メンバー数（5 → 10に増加）
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

        # Confidence Weighting（暗黙的フィードバック対応）
        'use_confidence_weighting': False,  # confidence weightingを使用するか（実験的機能）
        'confidence_alpha': 1.0,  # confidence = 1 + alpha * rating（0.5-2.0推奨）
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
        'use_cross_validation': True,  # 交差検証を使用（True推奨：汎化性能を評価）
        'n_folds': 3,  # 交差検証の分割数（3-5推奨、計算時間とのトレードオフ）
        'use_time_series_split': True,  # TimeSeriesSplitを使用（True推奨：時系列データでLook-ahead biasを防ぐ）
        'test_size': 0.15,  # Test setのサイズ（0.15 = 15%、チューニング時は触らない）
        'enable_early_stopping': True,  # Early stoppingを有効化（True推奨：計算時間短縮）
        'early_stopping_patience': 5,  # Early stopping待機回数（改善が見られない回数、3-5推奨）
        'early_stopping_batch_size': 50,  # Early stoppingのバッチサイズ（50=標準、100=高速、200=最速）
        'search_space': {
            'n_components': (10, 30),  # 探索範囲（最小, 最大）※40->30に縮小
            'alpha_W': (0.001, 0.5),  # 対数スケールで探索（正則化強度）※1.0->0.5に縮小
            'alpha_H': (0.001, 0.5),  # 対数スケールで探索（正則化強度）※1.0->0.5に縮小
            'l1_ratio': (0.0, 1.0),  # L1/L2のバランス
            'max_iter': (500, 1500),  # 最大イテレーション数※2000->1500に縮小（Early stoppingで自動短縮）
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

    # 評価パラメータ
    EVALUATION_PARAMS = {
        # 基本パラメータ
        'top_k': 10,  # 推薦数（評価時）
        'include_extended_metrics': True,  # MRR, MAP等の拡張メトリクスを計算するか
        'include_diversity_metrics': True,  # 多様性指標を計算するか

        # Hold-out評価
        'train_ratio': 0.8,  # 訓練データの割合
        'use_temporal_split': True,  # 時系列分割を使用するか（Falseの場合はランダム）

        # 交差検証
        'n_folds': 5,  # 交差検証の分割数
        'cv_use_temporal': True,  # 時系列交差検証を使用するか

        # Leave-One-Out評価
        'loo_max_users': None,  # 評価する最大ユーザー数（Noneの場合は全ユーザー）

        # ランダム分割パラメータ
        'min_test_items': 1,  # テストデータに必要な最小力量数

        # メトリクス計算オプション
        'calculate_gini': True,  # Gini Indexを計算するか
        'calculate_novelty': True,  # Noveltyを計算するか（要member_competence）

        # レポート出力
        'detailed_report': True,  # 詳細なレポートを出力するか
        'export_results': True,  # 結果をCSVに出力するか
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
