import datetime
import logging
import logging.config as cfg
import os
import sys

import yaml
import re

from elliot.utils.folder import build_log_folder


class TimeFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        record.time_filter = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        return True


def init(path_config, folder_log, log_level=logging.WARNING):
    # Pull in Logging Config
    path = os.path.join(path_config)
    build_log_folder(folder_log)
    folder_log = os.path.abspath(os.sep.join([folder_log, "elliot.log"]))
    pattern = re.compile('.*?\${(\w+)}.*?')
    loader = yaml.SafeLoader
    loader.add_implicit_resolver('!CUSTOM', pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', folder_log
                )
            return full_value
        return value

    loader.add_constructor('!CUSTOM', constructor_env_variables)

    with open(path, 'r') as stream:
        try:
            logging_config = yaml.load(stream, Loader=loader)
        except yaml.YAMLError as exc:
            print("Error Loading Logger Config")
            pass

    # Load Logging configs
    cfg.dictConfig(logging_config)

    # Initialize Log Levels
    log_level = log_level

    loggers = {name: logging.getLogger(name) for name in logging.root.manager.loggerDict}
    for _, log in loggers.items():
        log.setLevel(log_level)


def get_logger(name, log_level=logging.DEBUG):
    if name not in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
    else:
        logger = logging.root.manager.loggerDict[name]
    logger.setLevel(log_level)
    return logger


def get_logger_model(name, log_level=logging.DEBUG):
    logger = logging.root.manager.loggerDict[name]
    logger_es = logging.root.manager.loggerDict["EarlyStopping"]
    logger_es.addFilter(TimeFilter())
    logger_es.addHandler(logger.handlers[0])
    logger_es.setLevel(log_level)
    logger.setLevel(log_level)
    return logger


def prepare_logger(name, path, log_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.addFilter(TimeFilter())
    logger.setLevel(log_level)
    logfilepath = os.path.abspath(os.sep.join([path, f"{name}-{datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')}.log"]))
    fh = logging.FileHandler(logfilepath)
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(log_level)
    sh.setLevel(log_level)
    filefmt = "%(time_filter)-15s: %(levelname)-.1s %(message)s"
    # filedatefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(filefmt)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger