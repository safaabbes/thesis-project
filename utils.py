import logging
import re


def sorted_alphanumeric(data):
    '''
    https://gist.github.com/SeanSyue/8c8ff717681e9ecffc8e43a686e68fd9
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def get_logger(path_log):
    '''
    https://www.toptal.com/python/in-depth-python-logging
    '''

    # Get logger
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    # Get formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Get file handler and add it to logger
    fh = logging.FileHandler(path_log, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Get console handler and add it to logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False

    return logger