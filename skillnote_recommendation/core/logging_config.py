"""
構造化ロギング設定

GAFAレベルの構造化ログ出力をサポート
- JSON形式でログ出力
- トレースID対応
- 検索・集計可能なログ
"""

import logging
import sys
from typing import Any
import structlog
from pythonjsonlogger import jsonlogger


def setup_structured_logging(
    log_level: str = "INFO", enable_json: bool = True, enable_console: bool = True
) -> None:
    """
    構造化ロギングをセットアップ

    Args:
        log_level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        enable_json: JSON形式で出力するか
        enable_console: コンソールに出力するか
    """
    # 既存のログハンドラをクリア
    logging.root.handlers = []

    # ログレベル設定
    level = getattr(logging, log_level.upper(), logging.INFO)

    # structlogの設定
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 標準ライブラリのloggingも設定
    if enable_console:
        handler = logging.StreamHandler(sys.stdout)

        if enable_json:
            # JSON形式のフォーマッタ
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s", timestamp=True
            )
        else:
            # 人間可読形式
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

        handler.setFormatter(formatter)
        handler.setLevel(level)

        # ルートロガーに追加
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(level)


def get_logger(name: str) -> Any:
    """
    構造化ロガーを取得

    Args:
        name: ロガー名（通常は __name__ を指定）

    Returns:
        構造化ロガー

    Usage:
        logger = get_logger(__name__)
        logger.info("user_login", user_id="U123", ip="192.168.1.1")
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    ロガーをクラスに追加するMixin

    Usage:
        class MyClass(LoggerMixin):
            def some_method(self):
                self.logger.info("method_called", method="some_method")
    """

    @property
    def logger(self) -> Any:
        """構造化ロガーを取得"""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
        return self._logger
