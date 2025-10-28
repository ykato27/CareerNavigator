"""
推薦実行スクリプト

変換済みデータを使用して力量推薦を実行
"""

import logging
import sys
from skillnote_recommendation.core.recommendation_system import RecommendationSystem


logger = logging.getLogger(__name__)


def main():
    """メイン処理"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("スキルノート 推薦システム")
    logger.info("=" * 80)
    
    try:
        # 推薦システム初期化
        system = RecommendationSystem()
        
        # サンプル実行: 中程度の習得数のメンバーを選択
        member_stats = system.df_member_competence.groupby('メンバーコード').size().sort_values()
        sample_code = member_stats.index[len(member_stats)//2]
        
        logger.info("\n" + "=" * 80)
        logger.info("サンプル実行1: 全タイプの力量を推薦")
        logger.info("=" * 80)
        system.print_recommendations(sample_code, top_n=10)
        
        logger.info("\n" + "=" * 80)
        logger.info("サンプル実行2: SKILLタイプのみを推薦")
        logger.info("=" * 80)
        system.print_recommendations(sample_code, competence_type='SKILL', top_n=5)
        
        logger.info("\n" + "=" * 80)
        logger.info("処理完了")
        logger.info("=" * 80)
        logger.info("\nPythonコードからの利用方法:")
        logger.info("  from skillnote_recommendation import RecommendationSystem")
        logger.info("  system = RecommendationSystem()")
        logger.info("  system.print_recommendations('メンバーコード', top_n=10)")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error("\nエラー: 変換済みデータが見つかりません")
        logger.error("先に convert_data.py を実行してください")
        logger.error("\n  skillnote-convert")
        logger.error("  # または")
        logger.error("  python -m skillnote_recommendation.scripts.convert_data")
        return 1

    except Exception as e:
        logger.exception("\nエラー: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
