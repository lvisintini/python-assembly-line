import logging
from tqdm import tqdm

from production_line.data_helper import DataHelper


class LineHelper(object):
    tasks = []

    def __init__(self, data_helper=None, **kwargs):
        self.data_helper = DataHelper(**kwargs) if data_helper is None else data_helper
        if data_helper is not None:
            self.data_helper.__dict__.update(kwargs)

        logger = logging.getLogger(__name__)
        self.log = logger
        self.data_helper.log = logger

    def run(self):
        for task in tqdm([t for t in self.tasks if hasattr(t, 'setup')], desc='Setups'):
            self.log.debug('Setup for {!r}'.format(task))
            try:
                self.data_helper = task.do_setup(self.data_helper)
            except Exception as e:
                self.log.error('Setup for {!r}'.format(task))
                self.log.exception(e)
                break

        for task in tqdm([t for t in self.tasks if hasattr(t, 'process')], desc='Processes'):
            if hasattr(task, 'pre_process'):
                self.log.debug('Pre-Process for {!r}'.format(task))
                try:
                    self.data_helper = task.do_pre_process(self.data_helper)
                except Exception as e:
                    self.log.error('Pre-Process for {!r}'.format(task))
                    self.log.exception(e)
                    break

            self.log.debug('Process for {!r}'.format(task))
            try:
                self.data_helper = task.do_process(self.data_helper)
            except Exception as e:
                self.log.error('Error: Process for {!r}'.format(task))
                self.log.exception(e)
                break

            if hasattr(task, 'post_process'):
                self.log.debug('Post-Process for {!r}'.format(task))
                try:
                    self.data_helper = task.do_post_process(self.data_helper)
                except Exception as e:
                    self.log.error('Post-Process for {!r}'.format(task))
                    self.log.exception(e)
                    break

        for task in tqdm([t for t in self.tasks if hasattr(t, 'teardown')], desc='Teardowns'):
            self.log.debug('Teardown for {!r}'.format(task))
            try:
                self.data_helper = task.do_teardown(self.data_helper)
            except Exception as e:
                self.log.error('Teardown for {!r}'.format(task))
                self.log.exception(e)
                break



