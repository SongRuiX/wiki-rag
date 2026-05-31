import logging
import sys


def setup_logging(log_file: str = "./wiki-rag.log") -> None:
    """配置日志系统。控制台 INFO 级别，文件 DEBUG 级别。

    调用后，所有模块通过 logging.getLogger(__name__) 获取已配置的 logger。
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s %(funcName)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 handler: DEBUG 级别
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 控制台 handler: INFO 级别
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)
