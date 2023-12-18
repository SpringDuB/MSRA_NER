import logging
import os


def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设定日志的级别

    logger.handlers = []
    if save and (log_path is not None):
        log_path = os.path.abspath(log_path)
        if save and not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))

        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s: %(message)s'))
        logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(funcName)s:%(message)s'))
    logger.addHandler(stream_handler)
