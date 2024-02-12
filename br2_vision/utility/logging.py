import logging


def config_logging(verbose: bool):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def get_script_logger(name):
    logger = logging.getLogger(name)
    return logger
