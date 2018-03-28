import logging
from types import MethodType
from assembly_line.expections import AssemblyLineDataHelperNotReturnedError
from assembly_line.data_helper import DataHelper


logger = logging.getLogger(__name__)


def _do_pre_process(this, data_helper):
    if hasattr(this, 'pre_process'):
        res = this.pre_process(data_helper)

        if not isinstance(res, DataHelper):
            raise AssemblyLineDataHelperNotReturnedError(
                '{!r}.pre_process did not return a DataHelperInstance'.format(this)
            )

        return res
    return data_helper


def _do_process(this, data_helper):
    if not hasattr(this, 'process'):
        raise NotImplementedError('process method should be implemented for every Task class/instance')
    res = this.process(data_helper)

    if not isinstance(res, DataHelper):
        raise AssemblyLineDataHelperNotReturnedError(
            '{!r}.process did not return a DataHelperInstance'.format(this)
        )

    return res


def _do_post_process(this, data_helper):
    if hasattr(this, 'post_process'):
        res = this.post_process(data_helper)

        if not isinstance(res, DataHelper):
            raise AssemblyLineDataHelperNotReturnedError(
                '{!r}.post_process did not return a DataHelperInstance'.format(this)
            )

        return res
    return data_helper


def _do_setup(this, data_helper):
    if hasattr(this, 'setup'):
        res = this.setup(data_helper)

        if not isinstance(res, DataHelper):
            raise AssemblyLineDataHelperNotReturnedError(
                '{!r}.setup did not return a DataHelperInstance'.format(this)
            )

        return res
    return data_helper


def _do_teardown(this, data_helper):
    if hasattr(this, 'teardown'):
        res = this.teardown(data_helper)

        if not isinstance(res, DataHelper):
            raise AssemblyLineDataHelperNotReturnedError(
                '{!r}.teardown did not return a DataHelperInstance'.format(this)
            )

        return res
    return data_helper


class ReprMeta(type):
    def __call__(cls, *args, **kwargs):
        label = kwargs.pop('label', None)

        inst = super(ReprMeta, cls).__call__(*args, **kwargs)

        if not label:
            if not getattr(inst, 'label', None):
                if args and kwargs:
                    inst.label = f'{cls.__name__} args={args!r}, kwargs={kwargs!r}'
                elif args:
                    inst.label = f'{cls.__name__} args={args!r}'
                elif kwargs:
                    inst.label = f'{cls.__name__} kwargs={kwargs!r}'
                else:
                    inst.label = cls.__name__
        else:
            inst.label = label

        return inst


class Task(object, metaclass=ReprMeta):
    log = logger

    def __init__(self, *args, **kwargs):
        self.do_setup = MethodType(_do_setup, self)
        self.do_pre_process = MethodType(_do_pre_process, self)
        self.do_process = MethodType(_do_process, self)
        self.do_post_process = MethodType(_do_post_process, self)
        self.do_teardown = MethodType(_do_teardown, self)

    do_setup = classmethod(_do_setup)
    do_pre_process = classmethod(_do_pre_process)
    do_process = classmethod(_do_process)
    do_post_process = classmethod(_do_post_process)
    do_teardown = classmethod(_do_teardown)

    def __repr__(self):
        return getattr(self, 'label', super().__repr__())
