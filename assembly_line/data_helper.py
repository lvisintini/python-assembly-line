import logging

logger = logging.getLogger(__name__)


class DataHelper(object):
    log = logger

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
