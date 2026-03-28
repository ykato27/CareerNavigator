"""
データバリデーションの使用例

v2.0.0で追加されたPydanticベースのバリデーションのデモンストレーション
"""

from skillnote_recommendation.core.schemas import (
    RecommendationRequest,
    TopKPredictionRequest,
    EvaluationRequest,
    DataQualityReport
)
from pydantic import ValidationError


def example_valid_request():
    """正常なリクエストの検証"""
    print("=== 正常なリクエスト ===")

    # 推薦リクエスト
    request = RecommendationRequest(
        member_code="M001",
        top_n=10,
        competence_type=["SKILL", "EDUCATION"],
        diversity_strategy="hybrid"
    )

    print(f"✅ Valid Request:")
    print(f"   Member Code: {request.member_code}")
    print(f"   Top N: {request.top_n}")
    print(f"   Competence Type: {request.competence_type}")
    print(f"   Diversity Strategy: {request.diversity_strategy}")


def example_invalid_requests():
    """異常なリクエストの検証"""
    print("\n=== 異常なリクエストの検証 ===")

    # 1. 空のメンバーコード
    try:
        RecommendationRequest(
            member_code="",  # 空文字列
            top_n=10
        )
    except ValidationError as e:
        print(f"\n❌ Empty member code:")
        for error in e.errors():
            print(f"   {error['loc']}: {error['msg']}")

    # 2. 範囲外のtop_n
    try:
        RecommendationRequest(
            member_code="M001",
            top_n=1000  # 最大100まで
        )
    except ValidationError as e:
        print(f"\n❌ Top_n out of range:")
        for error in e.errors():
            print(f"   {error['loc']}: {error['msg']}")

    # 3. 不正なdiversity_strategy
    try:
        RecommendationRequest(
            member_code="M001",
            top_n=10,
            diversity_strategy="invalid"  # 'mmr', 'category', 'type', 'hybrid' のみ
        )
    except ValidationError as e:
        print(f"\n❌ Invalid diversity strategy:")
        for error in e.errors():
            print(f"   {error['loc']}: {error['msg']}")


def example_topk_prediction_request():
    """Top-K予測リクエストの検証"""
    print("\n=== Top-K予測リクエスト ===")

    request = TopKPredictionRequest(
        member_code="M001",
        k=20,
        exclude_acquired=True,
        acquired_competences=["C001", "C002", "C003"]
    )

    print(f"✅ Top-K Request:")
    print(f"   K: {request.k}")
    print(f"   Exclude Acquired: {request.exclude_acquired}")
    print(f"   Acquired Count: {len(request.acquired_competences) if request.acquired_competences else 0}")


def example_evaluation_request():
    """評価リクエストの検証"""
    print("\n=== 評価リクエスト ===")

    eval_request = EvaluationRequest(
        top_k=10,
        train_ratio=0.8,
        use_temporal_split=True,
        include_extended_metrics=True,
        include_diversity_metrics=True
    )

    print(f"✅ Evaluation Request:")
    print(f"   Top K: {eval_request.top_k}")
    print(f"   Train Ratio: {eval_request.train_ratio}")
    print(f"   Temporal Split: {eval_request.use_temporal_split}")


def example_data_quality_report():
    """データ品質レポートの使用"""
    print("\n=== データ品質レポート ===")

    # レポート作成
    report = DataQualityReport(is_valid=True)

    # エラー追加
    report.add_error("Found 5 NaN values")
    report.add_error("Matrix has negative values")

    # 警告追加
    report.add_warning("Matrix is 99% sparse")

    # 統計情報追加
    report.set_statistic("sparsity", 0.99)
    report.set_statistic("mean", 2.5)
    report.set_statistic("std", 0.8)

    # レポート確認
    print(f"Valid: {report.is_valid}")
    print(f"Errors: {report.errors}")
    print(f"Warnings: {report.warnings}")
    print(f"Statistics: {report.statistics}")


def example_request_to_dict():
    """リクエストを辞書に変換"""
    print("\n=== リクエストの辞書化 ===")

    request = RecommendationRequest(
        member_code="M001",
        top_n=10,
        diversity_strategy="hybrid"
    )

    # Pydantic の model_dump() を使用
    request_dict = request.model_dump()
    print(f"Request as dict: {request_dict}")


if __name__ == "__main__":
    example_valid_request()
    example_invalid_requests()
    example_topk_prediction_request()
    example_evaluation_request()
    example_data_quality_report()
    example_request_to_dict()
