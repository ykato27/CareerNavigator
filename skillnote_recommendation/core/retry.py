"""
リトライロジック

GAFAレベルのリトライ戦略:
- 指数バックオフ
- リトライ可能なエラーの自動判定
- 構造化ログ対応
"""

from typing import TypeVar, Callable, Any
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
    after_log,
)
import logging

from skillnote_recommendation.core.errors import RecommendationError


logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_retryable_error(exception: BaseException) -> bool:
    """
    エラーがリトライ可能かどうかを判定

    Args:
        exception: 発生した例外

    Returns:
        リトライ可能な場合True
    """
    if isinstance(exception, RecommendationError):
        return exception.retryable
    return False


def with_retry(
    max_attempts: int = 3,
    min_wait_seconds: int = 1,
    max_wait_seconds: int = 10,
    log_level: int = logging.WARNING,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    リトライデコレータ

    指数バックオフでリトライを実行する

    Args:
        max_attempts: 最大試行回数
        min_wait_seconds: 最小待機時間（秒）
        max_wait_seconds: 最大待機時間（秒）
        log_level: ログレベル

    Usage:
        @with_retry(max_attempts=3)
        def fetch_data():
            # リトライ可能なエラーの場合、最大3回リトライ
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait_seconds, max=max_wait_seconds),
            retry=retry_if_exception(is_retryable_error),
            before_sleep=before_sleep_log(logger, log_level),
            after=after_log(logger, log_level),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def with_retry_on_exception(
    exception_types: tuple[type[Exception], ...],
    max_attempts: int = 3,
    min_wait_seconds: int = 1,
    max_wait_seconds: int = 10,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    特定の例外でリトライするデコレータ

    Args:
        exception_types: リトライ対象の例外タプル
        max_attempts: 最大試行回数
        min_wait_seconds: 最小待機時間（秒）
        max_wait_seconds: 最大待機時間（秒）

    Usage:
        @with_retry_on_exception((NetworkError, TimeoutError))
        def fetch_from_api():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait_seconds, max=max_wait_seconds),
            retry=retry_if_exception(lambda e: isinstance(e, exception_types)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        return wrapper

    return decorator
