import logging
from datetime import datetime
import os
import sys

def get_strtime(create_time:datetime):
    return create_time.strftime('%Y.%m.%d-%H.%M.%S') if create_time else datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

def getLogger(name=None, path=None, create_time:datetime=None, console_level=logging.DEBUG, file_level=logging.DEBUG):
    path = os.path.abspath(path or 'logs')
    os.makedirs(path, exist_ok=True)
    file_log = os.path.join(path, f'{name}-{get_strtime(create_time)}.log')
    with open(file_log, 'w') as f:
            pass
    logfile_handler = logging.FileHandler(file_log, mode='a', encoding="utf8")
    logfile_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] TID.%(thread)d %(module)s.%(lineno)d %(name)s:\t%(message)s'))
    logfile_handler.setLevel(file_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',datefmt="%Y/%m/%d %H:%M:%S"))
    console_handler.setLevel(console_level)
    logger = logging.getLogger(name)
    logger.addHandler(logfile_handler)
    logger.addHandler(console_handler)
    logger.setLevel(min(console_level, file_level))
    return logger
