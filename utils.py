import logging
import re


def deprocess(image):
    # Remove the normalization
    image = image.clone()
    image[0] = image[0]*0.229 + 0.485
    image[1] = image[1]*0.224 + 0.456
    image[2] = image[2]*0.225 + 0.406
    # Convert the image from tensor to numpy
    image = image.cpu().numpy()
    # Transpose the image from (C, H, W) to (H, W, C)
    image = image.transpose((1, 2, 0))
    return image

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