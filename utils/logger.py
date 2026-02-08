#!/usr/bin/env python3
"""日志模块，提供带颜色的终端输出和文件日志记录功能。"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class LogColor:
    """日志ANSI颜色常量"""

    INFO = "\033[32m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """自定义Formatter，为不同级别的日志添加颜色"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录并添加颜色。

        Args:
            record: 日志记录对象

        Returns:
            格式化并着色后的日志字符串
        """
        if hasattr(record, "caller_info"):
            filename, lineno = record.caller_info
            basename = Path(filename).name
            prefix = f"[{record.levelname}][{basename}:{lineno}]"
        else:
            basename = Path(record.filename).name
            prefix = f"[{record.levelname}][{basename}:{record.lineno}]"

        message = record.getMessage()
        formatted = f"{prefix} {message}"

        if record.levelno == logging.INFO:
            return f"{LogColor.INFO}{formatted}{LogColor.RESET}"
        elif record.levelno == logging.WARNING:
            return f"{LogColor.WARNING}{formatted}{LogColor.RESET}"
        elif record.levelno == logging.ERROR:
            return f"{LogColor.ERROR}{formatted}{LogColor.RESET}"

        return formatted


class Logger:
    """日志器类，封装标准logging模块。"""

    def __init__(self, name: str, log_dir: str = ".cache/logs") -> None:
        """初始化日志器。

        Args:
            name: 日志器名称
            log_dir: 日志文件目录
        """
        self._name = name
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(ColorFormatter())

            log_file = self._log_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "[%(levelname)s][%(filename)s:%(lineno)d] %(message)s"
            )
            file_handler.setFormatter(file_formatter)

            self._logger.addHandler(console_handler)
            self._logger.addHandler(file_handler)

    def _log(self, level: int, message: str) -> None:
        """记录日志。

        Args:
            level: 日志级别
            message: 日志消息
        """
        caller_frame = sys._getframe(2)
        caller_filename = caller_frame.f_code.co_filename
        caller_lineno = caller_frame.f_lineno

        record = self._logger.makeRecord(
            self._name, level, caller_filename, caller_lineno, message, (), None
        )
        record.caller_info = (caller_filename, caller_lineno)

        self._logger.handle(record)

    def info(self, message: str) -> None:
        """输出info级别日志。

        Args:
            message: 日志消息
        """
        self._log(logging.INFO, message)

    def warning(self, message: str) -> None:
        """输出warning级别日志。

        Args:
            message: 日志消息
        """
        self._log(logging.WARNING, message)

    def error(self, message: str) -> None:
        """输出error级别日志。

        Args:
            message: 日志消息
        """
        self._log(logging.ERROR, message)


_loggers: dict[str, Logger] = {}


def get_logger(name: str = "quant_factor") -> Logger:
    """获取日志器实例。

    Args:
        name: 日志器名称

    Returns:
        Logger实例
    """
    if name not in _loggers:
        _loggers[name] = Logger(name)
    return _loggers[name]


def info(message: str) -> None:
    """输出info级别日志。

    Args:
        message: 日志消息
    """
    get_logger().info(message)


def warning(message: str) -> None:
    """输出warning级别日志。

    Args:
        message: 日志消息
    """
    get_logger().warning(message)


def error(message: str) -> None:
    """输出error级别日志。

    Args:
        message: 日志消息
    """
    get_logger().error(message)
