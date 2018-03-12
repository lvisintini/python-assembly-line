import logging
from colorlog import ColoredFormatter
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = TqdmHandler()
ch.setLevel(logging.DEBUG)

color_formatter = ColoredFormatter(
    datefmt='%y-%m-%d %H;%M:%S',
    log_colors={
        'DEBUG': 'blue',
        'INFO': 'cyan',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
ch.setFormatter(color_formatter)

logger.addHandler(ch)
