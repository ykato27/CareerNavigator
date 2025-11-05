"""
構造化ロギングの基本的な使用例

v2.0.0で追加された構造化ロギング機能のデモンストレーション
"""

from skillnote_recommendation.core.logging_config import (
    setup_structured_logging,
    get_logger,
    LoggerMixin
)


def example_basic_logging():
    """基本的なロギング"""
    # ロギング設定（開発環境: 人間可読形式）
    setup_structured_logging(
        log_level="INFO",
        enable_json=False  # False=人間可読、True=JSON
    )

    logger = get_logger(__name__)

    # 構造化ログの出力
    logger.info(
        "user_login",
        user_id="U123",
        ip_address="192.168.1.1",
        login_method="oauth"
    )

    logger.warning(
        "slow_query_detected",
        query_time_ms=1500,
        threshold_ms=1000,
        query="SELECT * FROM members"
    )

    logger.error(
        "recommendation_failed",
        member_code="M001",
        error_code="R001",
        error_message="Cold start problem"
    )


def example_logger_mixin():
    """LoggerMixinの使用例"""

    class MyRecommender(LoggerMixin):
        def recommend(self, member_code: str, top_n: int = 10):
            self.logger.info(
                "recommendation_started",
                member_code=member_code,
                top_n=top_n
            )

            # ... 推薦処理 ...

            self.logger.info(
                "recommendation_completed",
                member_code=member_code,
                recommendations_count=top_n,
                duration_ms=123
            )

    recommender = MyRecommender()
    recommender.recommend("M001", 10)


def example_json_logging():
    """JSON形式のロギング（本番環境向け）"""
    setup_structured_logging(
        log_level="INFO",
        enable_json=True  # JSON形式で出力
    )

    logger = get_logger(__name__)

    logger.info(
        "recommendation_generated",
        member_code="M001",
        recommendations=[
            {"code": "C001", "score": 0.95},
            {"code": "C002", "score": 0.87}
        ],
        diversity_metrics={
            "category_diversity": 0.75,
            "type_diversity": 0.50
        }
    )
    # 出力: JSON形式で検索・集計可能


if __name__ == "__main__":
    print("=== 基本的なロギング ===")
    example_basic_logging()

    print("\n=== LoggerMixinの使用 ===")
    example_logger_mixin()

    print("\n=== JSON形式のロギング ===")
    example_json_logging()
