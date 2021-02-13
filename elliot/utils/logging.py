import datetime
import logging
import logging.config as cfg
import os
import yaml
import re

from elliot.utils.folder import build_log_folder


def init(path_config, folder_log, log_level=logging.WARNING):
    # Pull in Logging Config
    path = os.path.join(path_config)
    build_log_folder(folder_log)
    folder_log = f'{folder_log}elliot.log'
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


def prepare_logger(name, path, log_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logfilepath = f'''{path}/{name}-{datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')}.log'''
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(log_level)
    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(filefmt, filedatefmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
