"""
推薦システムのエラー定義

GAFAレベルのエラーハンドリング:
- エラーコード体系
- 構造化されたエラーコンテキスト
- リトライ可能性の判定
"""

from enum import Enum
from typing import Any, Optional


class ErrorCode(Enum):
    """エラーコード定義"""

    # データ関連 (D001-D099)
    DATA_NOT_FOUND = "D001"
    DATA_INVALID = "D002"
    DATA_INCONSISTENCY = "D003"
    DATA_QUALITY_ERROR = "D004"

    # モデル関連 (M001-M099)
    MODEL_NOT_TRAINED = "M001"
    MODEL_LOAD_ERROR = "M002"
    MODEL_PREDICTION_ERROR = "M003"
    MODEL_SAVE_ERROR = "M004"

    # 推薦関連 (R001-R099)
    COLD_START = "R001"
    INVALID_PARAMETER = "R002"
    RECOMMENDATION_FAILED = "R003"

    # 外部サービス関連 (E001-E099)
    EXTERNAL_SERVICE_ERROR = "E001"
    NETWORK_ERROR = "E002"
    TIMEOUT_ERROR = "E003"

    # システム関連 (S001-S099)
    CONFIGURATION_ERROR = "S001"
    INTERNAL_ERROR = "S002"
    RESOURCE_EXHAUSTED = "S003"


class RecommendationError(Exception):
    """
    推薦システムの基底例外クラス

    全てのカスタム例外はこのクラスを継承する
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        retryable: bool = False,
        **context: Any
    ):
        """
        Args:
            code: エラーコード
            message: エラーメッセージ
            retryable: リトライ可能なエラーか
            **context: エラーコンテキスト（追加情報）
        """
        self.code = code
        self.message = message
        self.retryable = retryable
        self.context = context
        super().__init__(f"[{code.value}] {message}")

    def to_dict(self) -> dict[str, Any]:
        """エラー情報を辞書形式で返す"""
        return {
            'error_code': self.code.value,
            'message': self.message,
            'retryable': self.retryable,
            'context': self.context
        }


# =============================================================================
# データ関連エラー
# =============================================================================

class DataNotFoundError(RecommendationError):
    """データが見つからない"""

    def __init__(self, resource: str, identifier: str, **context: Any):
        super().__init__(
            ErrorCode.DATA_NOT_FOUND,
            f"{resource} not found: {identifier}",
            retryable=False,
            resource=resource,
            identifier=identifier,
            **context
        )


class DataInvalidError(RecommendationError):
    """データが不正"""

    def __init__(self, message: str, **context: Any):
        super().__init__(
            ErrorCode.DATA_INVALID,
            message,
            retryable=False,
            **context
        )


class DataQualityError(RecommendationError):
    """データ品質エラー"""

    def __init__(self, message: str, quality_issues: list[str], **context: Any):
        super().__init__(
            ErrorCode.DATA_QUALITY_ERROR,
            message,
            retryable=False,
            quality_issues=quality_issues,
            **context
        )


# =============================================================================
# モデル関連エラー
# =============================================================================

class ModelNotTrainedError(RecommendationError):
    """モデルが未学習"""

    def __init__(self, model_type: Optional[str] = None):
        super().__init__(
            ErrorCode.MODEL_NOT_TRAINED,
            "Model is not trained. Please train the model first.",
            retryable=False,
            model_type=model_type,
            suggestion="Call fit() or load() before making predictions"
        )


class ModelLoadError(RecommendationError):
    """モデルの読み込みエラー"""

    def __init__(self, filepath: str, reason: str):
        super().__init__(
            ErrorCode.MODEL_LOAD_ERROR,
            f"Failed to load model from {filepath}: {reason}",
            retryable=True,
            filepath=filepath,
            reason=reason
        )


class ModelPredictionError(RecommendationError):
    """予測実行エラー"""

    def __init__(self, message: str, **context: Any):
        super().__init__(
            ErrorCode.MODEL_PREDICTION_ERROR,
            message,
            retryable=False,
            **context
        )


# =============================================================================
# 推薦関連エラー
# =============================================================================

class ColdStartError(RecommendationError):
    """
    コールドスタート問題

    新規ユーザーなど、学習データに存在しないメンバーに対する推薦エラー
    """

    def __init__(self, member_code: str, **context: Any):
        super().__init__(
            ErrorCode.COLD_START,
            f"Cold start problem: Member '{member_code}' not found in training data",
            retryable=False,
            member_code=member_code,
            suggestion="Add member competence data or use content-based fallback",
            **context
        )


class InvalidParameterError(RecommendationError):
    """パラメータが不正"""

    def __init__(self, parameter: str, value: Any, reason: str):
        super().__init__(
            ErrorCode.INVALID_PARAMETER,
            f"Invalid parameter '{parameter}': {reason}",
            retryable=False,
            parameter=parameter,
            value=str(value),
            reason=reason
        )


class RecommendationFailedError(RecommendationError):
    """推薦の生成に失敗"""

    def __init__(self, message: str, **context: Any):
        super().__init__(
            ErrorCode.RECOMMENDATION_FAILED,
            message,
            retryable=True,
            **context
        )


# =============================================================================
# 外部サービス関連エラー
# =============================================================================

class ExternalServiceError(RecommendationError):
    """外部サービスエラー"""

    def __init__(self, service: str, message: str, **context: Any):
        super().__init__(
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            f"External service error ({service}): {message}",
            retryable=True,
            service=service,
            **context
        )


class NetworkError(RecommendationError):
    """ネットワークエラー"""

    def __init__(self, message: str, **context: Any):
        super().__init__(
            ErrorCode.NETWORK_ERROR,
            message,
            retryable=True,
            **context
        )


# =============================================================================
# システム関連エラー
# =============================================================================

class ConfigurationError(RecommendationError):
    """設定エラー"""

    def __init__(self, message: str, **context: Any):
        super().__init__(
            ErrorCode.CONFIGURATION_ERROR,
            message,
            retryable=False,
            **context
        )


# =============================================================================
# 後方互換性のための旧例外クラス（非推奨）
# =============================================================================

class MLModelNotTrainedError(ModelNotTrainedError):
    """
    非推奨: ModelNotTrainedError を使用してください

    後方互換性のために残していますが、将来的に削除される可能性があります。
    """
    pass
