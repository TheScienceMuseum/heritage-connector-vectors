import logging
import sys


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


class DisableLogger:
    """A context manager to temporarily disable logging. I.e.
    ```
    with DisableLogger():
        # do things and don't shout about them
    ```
    """

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)
