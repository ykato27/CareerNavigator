"""
エラーハンドリングの使用例

v2.0.0で追加された構造化エラー処理のデモンストレーション
"""

from skillnote_recommendation.core.errors import (
    ColdStartError,
    ModelNotTrainedError,
    InvalidParameterError,
    DataNotFoundError,
    RecommendationError,
    ErrorCode
)
from skillnote_recommendation.core.retry import with_retry, with_retry_on_exception


def example_error_raising():
    """エラーの発生"""

    # コールドスタートエラー
    try:
        member_code = "NEW_MEMBER"
        # 新規メンバーのチェック
        if member_code not in ["M001", "M002"]:
            raise ColdStartError(
                member_code=member_code,
                suggestion="Add member data first"
            )
    except ColdStartError as e:
        print(f"❌ Error: {e}")
        print(f"   Code: {e.code.value}")
        print(f"   Retryable: {e.retryable}")
        print(f"   Context: {e.context}")

    # パラメータエラー
    try:
        top_n = -5
        if top_n <= 0:
            raise InvalidParameterError(
                parameter="top_n",
                value=top_n,
                reason="Must be positive integer"
            )
    except InvalidParameterError as e:
        print(f"\n❌ Parameter Error: {e}")
        print(f"   Parameter: {e.context['parameter']}")
        print(f"   Invalid Value: {e.context['value']}")


def example_error_dict():
    """エラー情報を辞書形式で取得"""
    try:
        raise DataNotFoundError(
            resource="Member",
            identifier="M999",
            search_criteria={"department": "Engineering"}
        )
    except DataNotFoundError as e:
        error_dict = e.to_dict()
        print("\n📄 Error Dictionary:")
        print(f"   Error Code: {error_dict['error_code']}")
        print(f"   Message: {error_dict['message']}")
        print(f"   Retryable: {error_dict['retryable']}")
        print(f"   Context: {error_dict['context']}")


@with_retry(max_attempts=3, min_wait_seconds=1, max_wait_seconds=5)
def example_function_with_retry():
    """リトライ可能な関数"""
    print("⏳ Attempting operation...")

    # リトライ可能なエラーをシミュレート
    import random
    if random.random() < 0.7:  # 70%の確率でエラー
        raise RecommendationError(
            code=ErrorCode.RECOMMENDATION_FAILED,
            message="Temporary failure",
            retryable=True  # リトライ可能
        )

    print("✅ Operation succeeded!")
    return "Success"


def example_retry_logic():
    """リトライロジックの使用例"""
    try:
        result = example_function_with_retry()
        print(f"Result: {result}")
    except RecommendationError as e:
        print(f"❌ Final failure after retries: {e}")


if __name__ == "__main__":
    print("=== エラーの発生 ===")
    example_error_raising()

    print("\n=== エラー辞書 ===")
    example_error_dict()

    print("\n=== リトライロジック ===")
    example_retry_logic()
