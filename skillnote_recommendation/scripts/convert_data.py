"""
"""
データ変換スクリプト

CSVファイルを読み込み、推薦システム用にデータを変換
"""

import logging
import sys
from skillnote_recommendation.core.config import Config
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.core.similarity_calculator import SimilarityCalculator


logger = logging.getLogger(__name__)


def main():
    """メイン処理"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("スキルノート データ変換処理")
    logger.info("=" * 80)
    logger.info("\n入力ディレクトリ: %s", Config.DATA_DIR)
    logger.info("出力ディレクトリ: %s", Config.OUTPUT_DIR)
    
    # ディレクトリ作成
    Config.ensure_directories()
    
    try:
        # ステップ1: データ読み込み
        loader = DataLoader()
        data = loader.load_all_data()
        
        # データ検証
        if not loader.validate_data(data):
            logger.error("\nエラー: データ検証に失敗しました")
            return 1
        
        # ステップ2: データ変換
        transformer = DataTransformer()
        
        # 統合力量マスタ作成
        competence_master = transformer.create_competence_master(data)
        
        # メンバー習得力量データ作成
        member_competence, valid_members = transformer.create_member_competence(
            data, competence_master
        )
        
        # メンバー×力量マトリクス作成
        skill_matrix = transformer.create_skill_matrix(member_competence)
        
        # メンバーマスタクリーニング
        members_clean = transformer.clean_members_data(data)
        
        # ステップ3: 類似度計算
        similarity_calc = SimilarityCalculator()
        similarity_df = similarity_calc.calculate_similarity(member_competence)
        
        # ステップ4: データ保存
        logger.info("\n" + "=" * 80)
        logger.info("データ保存")
        logger.info("=" * 80)
        
        members_clean.to_csv(
            Config.get_output_path('members_clean'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        logger.info("  ✓ %s: %d件", Config.OUTPUT_FILES['members_clean'], len(members_clean))
        
        competence_master.to_csv(
            Config.get_output_path('competence_master'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        logger.info(
            "  ✓ %s: %d件",
            Config.OUTPUT_FILES['competence_master'],
            len(competence_master),
        )
        
        member_competence.to_csv(
            Config.get_output_path('member_competence'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        logger.info(
            "  ✓ %s: %d件",
            Config.OUTPUT_FILES['member_competence'],
            len(member_competence),
        )
        
        skill_matrix.to_csv(
            Config.get_output_path('skill_matrix'),
            encoding=Config.OUTPUT_ENCODING
        )
        logger.info(
            "  ✓ %s: %d×%d",
            Config.OUTPUT_FILES['skill_matrix'],
            skill_matrix.shape[0],
            skill_matrix.shape[1],
        )
        
        similarity_df.to_csv(
            Config.get_output_path('competence_similarity'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        logger.info(
            "  ✓ %s: %d件",
            Config.OUTPUT_FILES['competence_similarity'],
            len(similarity_df),
        )

        logger.info("\n" + "=" * 80)
        logger.info("データ変換処理完了")
        logger.info("=" * 80)
        logger.info("\n変換データは %s/ に保存されました", Config.OUTPUT_DIR)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error("\nエラー: %s", e)
        logger.error("\n必要なCSVファイルを %s/ に配置してください:", Config.DATA_DIR)
        for key, filename in Config.INPUT_FILES.items():
            logger.error("  - %s", filename)
        return 1

    except Exception as e:
        logger.exception("\nエラー: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
