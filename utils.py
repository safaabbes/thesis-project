import logging

def setup_logger(file_name=None, logger_name = __name__): 
        
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers.clear()
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter) 
    logger.addHandler(sh)
    
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
       
     
    return logger