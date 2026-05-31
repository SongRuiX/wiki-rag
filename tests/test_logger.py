import logging
import os
import tempfile
from src.logger import setup_logging


def test_setup_logging_creates_file_handler():
    """setup_logging should add a file handler and a console handler."""
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "test.log")
        root = logging.getLogger()

        initial_file_count = len([h for h in root.handlers if isinstance(h, logging.FileHandler)])
        initial_stream_count = len([h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)])
        initial_total = len(root.handlers)

        new_file_handler = None
        new_stream_handler = None
        try:
            setup_logging(log_path)

            # 断言：新增了两个 handler（文件 + 控制台）
            assert len(root.handlers) == initial_total + 2

            # 断言：文件 handler 增加了 1，且能找到指向 log_path 的那个
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == initial_file_count + 1
            matching = [h for h in file_handlers if h.baseFilename == os.path.abspath(log_path)]
            assert len(matching) == 1
            new_file_handler = matching[0]

            # 断言：StreamHandler (非 FileHandler) 增加了 1
            stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
            assert len(stream_handlers) == initial_stream_count + 1
            new_stream_handler = stream_handlers[-1]

            # 断言：文件 handler 级别为 DEBUG
            assert new_file_handler.level == logging.DEBUG

            # 断言：控制台 handler 级别为 INFO
            assert new_stream_handler.level == logging.INFO
        finally:
            # 清理新增的 handler
            if new_file_handler is not None:
                new_file_handler.close()
                root.removeHandler(new_file_handler)
            if new_stream_handler is not None:
                new_stream_handler.close()
                root.removeHandler(new_stream_handler)


def test_setup_logging_actually_writes_to_file():
    """日志消息应实际写入文件。"""
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "test.log")
        root = logging.getLogger()

        added = []
        try:
            setup_logging(log_path)
            # Track which handlers we added (the last 2)
            added = list(root.handlers)[-2:]

            logger = logging.getLogger("test_writer")
            logger.info("hello world")
            logger.debug("debug detail")

            # 刷新 handler
            for h in root.handlers:
                h.flush()

            with open(log_path, encoding="utf-8") as f:
                content = f.read()

            assert "hello world" in content
            assert "debug detail" in content
            assert "INFO" in content
            assert "DEBUG" in content
            assert "test_writer" in content
        finally:
            for h in added:
                h.close()
                root.removeHandler(h)


def test_setup_logging_format():
    """验证日志格式包含时间、级别、模块名、函数名、行号、消息。"""
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "test.log")
        root = logging.getLogger()

        added = []
        try:
            setup_logging(log_path)
            added = list(root.handlers)[-2:]

            logger = logging.getLogger("src.sync_strategy")

            def dummy_sync():
                logger.info("同步完成")

            dummy_sync()

            for h in root.handlers:
                h.flush()

            with open(log_path, encoding="utf-8") as f:
                content = f.read()

            # 格式: YYYY-MM-DD HH:MM:SS LEVEL NAME funcName:lineno message
            assert "INFO" in content
            assert "src.sync_strategy" in content
            assert "dummy_sync" in content
            assert "同步完成" in content
        finally:
            for h in added:
                h.close()
                root.removeHandler(h)
