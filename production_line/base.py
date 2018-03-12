import hashlib
import json
import os
import readline
from collections import OrderedDict, defaultdict

from production_line.tasks import Task


class LoadData(Task):
    sources = []
    extension = 'json'
    attr = 'data'

    def __init__(self, root, sources=None, extension=None, attr=None):
        super(LoadData, self).__init__()
        self.root = root
        self.sources = sources or self.sources
        self.extension = extension or self.extension
        self.attr = attr or self.attr

    def process(self, data_helper):
        data = OrderedDict()
        for source_key in self.sources:
            with open(f'{self.root}{source_key}.{self.extension}', 'r') as file_object:
                data[source_key] = json.load(file_object, object_pairs_hook=OrderedDict)
            self.log.info(f'{source_key} data loaded')
        setattr(data_helper, self.attr, data)
        return data_helper


class SaveData(Task):
    sources = []
    extension = 'json'
    indent = 2
    attr = 'data'

    def __init__(self, root, sources=None, extension=None, indent=None, attr=None):
        super(SaveData, self).__init__()
        self.root = root
        self.sources = sources or self.sources
        self.extension = extension or self.extension
        self.indent = indent or self.indent
        self.attr = attr or self.attr

    def process(self, data_helper):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for source_key in self.sources:
            with open(f'{self.root}{source_key}.{self.extension}', 'w') as file_object:
                json.dump(
                    getattr(data_helper, self.attr)[source_key],
                    file_object,
                    indent=self.indent,
                    ensure_ascii=False
                )
        return data_helper


class AppendData(Task):
    sources = []
    extension = 'json'
    indent = 2
    attr = 'data'

    def __init__(self, root, sources=None, extension=None, indent=None, attr=None):
        super(AppendData, self).__init__()
        self.root = root
        self.sources = sources or self.sources
        self.extension = extension or self.extension
        self.indent = indent or self.indent
        self.attr = attr or self.attr

    def process(self, data_helper):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for source_key in self.sources:
            abs_path = f'{self.root}{source_key}.{self.extension}'
            data = []
            if os.path.exists(abs_path):
                with open(abs_path, 'r') as file_object:
                    data = json.load(file_object)

            with open(abs_path, 'w') as file_object:
                json.dump(
                    data + getattr(data_helper, self.attr)[source_key],
                    file_object,
                    indent=self.indent,
                    ensure_ascii=False
                )
        return data_helper


class DataCollector(Task):
    pk = 'id'
    field_name = None
    amend_data = False
    use_memory = True
    null_input = '.'
    skip_input = ''

    def __init__(self, source=None, field_name=None, pk=None, amend_data=None, use_memory=None, null_input=None,
                 skip_input=None):
        super(DataCollector, self).__init__()
        self.source = source or self.source
        self.pk = pk or self.pk
        self.field_name = field_name or self.field_name
        self.amend_data = amend_data if amend_data is not None else self.amend_data
        self.use_memory = use_memory if use_memory is not None else self.use_memory

        if null_input not in [None, False]:
            self.null_input = null_input
        elif null_input is False:
            self.null_input = None

        if skip_input not in [None, False]:
            self.skip_input = skip_input
        elif skip_input is False:
            self.skip_input = None

    def before_each(self, model, data_helper):
        pass

    def after_each(self, model, data_helper):
        pass

    def input_text(self, model):
        template = '\n{desc}\nWhich {field!r} should it have?\nResponse{extra}: '

        desc = '{name!r}'.format(**model) if 'id' not in model else '({id!r}) {name!r}'.format(**model)
        extra = []
        if self.skip_input is not None:
            extra.append(f'{self.skip_input!r} to skip')
        if self.null_input is not None:
            extra.append(f'{self.null_input!r} for null')

        return template.format(
            desc=desc,
            field=self.field_name,
            extra=f' ({"; ".join(extra)})' if extra else ''
        )

    def clean_input(self, new_data):
        raise NotImplementedError

    def validate_input(self, new_data):
        raise NotImplementedError

    def load_from_memory(self, data_helper, model):
        if hasattr(data_helper, 'memory'):
            if self.pk in model and model[self.pk] in data_helper.memory[self.source][self.field_name]:
                model[self.field_name] = data_helper.memory[self.source][self.field_name][model[self.pk]]
                return model
        return model

    def pre_process_existing_value_to_prefill_input(self, value):
        if self.null_input is not None and value is None:
            return self.null_input
        return '' if value is None else str(value)

    def prompt_user(self, model, value, existing):
        def hook():
            readline.insert_text(self.pre_process_existing_value_to_prefill_input(value))
            readline.redisplay()
        if existing:
            readline.set_pre_input_hook(hook)
        result = input(self.input_text(model))
        readline.set_pre_input_hook()
        return result

    def handle_input_loop(self, model):
        existing_data = self.field_name in model
        new_data = model.get(self.field_name, None)

        first = True

        while first or not self.validate_input(new_data):
            if not first:
                print('No. That value is not right!. Try again...')
            else:
                first = False

            new_data = self.prompt_user(model, new_data, existing_data)

            if self.skip_input is not None and new_data == self.skip_input:
                return False, None

            new_data = self.clean_input(new_data)

        else:
            return True, new_data

    def process(self, data_helper):
        try:
            for model in data_helper.data[self.source]:
                if not self.amend_data:
                    if self.use_memory:
                        self.load_from_memory(data_helper, model)

                    if (self.field_name in model) and (self.validate_input(model[self.field_name])):
                        continue
                else:
                    if self.use_memory:
                        self.load_from_memory(data_helper, model)

                self.before_each(model, data_helper)

                success, new_data = self.handle_input_loop(model)

                if success:
                    model[self.field_name] = new_data
                    if self.use_memory and hasattr(data_helper, 'memory'):
                        data_helper.memory[self.source][self.field_name][model[self.pk]] = new_data
                self.after_each(model, data_helper)

        except (KeyboardInterrupt, SystemExit):
            print('')
            data_helper.log.warning('Exiting data gathering ...')
        finally:
            data_helper.log.info('Done collecting data for {}.{}!!'.format(self.source, self.field_name))
        return data_helper


class IntegerDataCollector(DataCollector):

    def clean_input(self, new_data):
        if self.null_input is not None and new_data == self.null_input:
            return None
        try:
            return int(new_data)
        except ValueError:
            return new_data

    def validate_input(self, new_data):
        return isinstance(new_data, int) or (self.null_input is not None and new_data is None)


class TextDataCollector(DataCollector):

    def clean_input(self, new_data):
        if self.null_input is not None and new_data == self.null_input:
            return None
        new_data = new_data.strip()
        return new_data

    def validate_input(self, new_data):
        return isinstance(new_data, str) or (self.null_input is not None and new_data is None)


class ChoiceDataCollector(DataCollector):
    choices = []

    def __init__(self, choices=None, **kwargs):
        self.choices = choices or self.choices

        super(ChoiceDataCollector, self).__init__(**kwargs)

    def pre_process_existing_value_to_prefill_input(self, value):
        if self.null_input is not None and value is None:
            return self.null_input
        return next(i for i in range(len(self.choices)) if self.choices[i][0] == value)

    def clean_input(self, new_data):
        if self.null_input is not None and new_data == self.null_input:
            return None
        if new_data.isdigit() and int(new_data) in range(len(self.choices)):
            return self.choices[int(new_data)][0]
        else:
            return new_data

    def validate_input(self, new_data):
        return new_data in dict(self.choices).keys() or (self.null_input is not None and new_data is None)

    def input_text(self, model):
        template = '\n{desc}\nWhich {field!r} should it have?\n\t{options_text}\nResponse{extra}: '

        desc = '{name!r}'.format(**model) if 'id' not in model else '({id!r}) {name!r}'.format(**model)

        extra = []
        if self.skip_input is None:
            extra.append(f'{self.skip_inputs!r} to skip')
        if self.null_input is not None:
            extra.append(f'{self.null_input!r} for null')

        options = []
        for i in range(len(self.choices)):
            v = f'{i} - {self.choices[i][1]} [{self.choices[i][0]}]'
            if self.choices[i][0] == self.choices[i][1]:
                v = f'{i} - {self.choices[i][1]}'
            options.append(v)

        options_text = '\n\t'.join(options)

        return template.format(
            desc=desc,
            field=self.field_name,
            options_text=options_text,
            extra=f' ({"; ".join(extra)})' if extra else ''
        )


class BooleanChoiceDataCollector(ChoiceDataCollector):
    choices = [
        (False, 'False'),
        (True, 'True'),
    ]

    def __init__(self, **kwargs):
        if 'choices' in kwargs:
            self.log.warning('You should not be providing choices for a BooleanChoiceDataCollector Task')
            kwargs.pop('choices')
        super(BooleanChoiceDataCollector, self).__init__(**kwargs)


class AppendChoiceDataCollector(ChoiceDataCollector):
    done_input = '*'
    accept_duplicates = False

    def __init__(self, done_input=None, accept_duplicates=None, **kwargs):
        self.done_input = done_input if done_input is not None else self.done_input
        self.accept_duplicates = bool(accept_duplicates) if accept_duplicates is not None else self.accept_duplicates
        super(AppendChoiceDataCollector, self).__init__(**kwargs)

    def pre_process_existing_value_to_prefill_input(self, value):
        return ''

    def validate_input(self, new_data):
        if isinstance(new_data, list) or isinstance(new_data, tuple):
            return all([
                super(AppendChoiceDataCollector, self).validate_input(x)
                for x in new_data
            ])

        if new_data is None:
            return self.null_input is not None

        return super().validate_input(new_data)

    def input_text(self, model, collected):
        template = '\n{desc}\nField {field!r} has {collected!r}. What else should it have?\n\t' \
                   '{options_text}\nResponse{extra}: '

        desc = '{name!r}'.format(**model) if 'id' not in model else '({id!r}) {name!r}'.format(**model)

        extra = []
        if self.skip_input is None:
            extra.append(f'{self.skip_inputs!r} to skip')
        if self.null_input is not None:
            extra.append(f'{self.null_input!r} for null')
        if self.done_input is not None:
            extra.append(f'{self.done_input!r} if done')

        options = []
        for i in range(len(self.choices)):
            v = f'{i} - {self.choices[i][1]} [{self.choices[i][0]}]'
            if self.choices[i][0] == self.choices[i][1]:
                v = f'{i} - {self.choices[i][1]}'
            options.append(v)

        options_text = '\n\t'.join(options)

        return template.format(
            desc=desc,
            field=self.field_name,
            collected=collected,
            options_text=options_text,
            extra=f' ({"; ".join(extra)})' if extra else ''
        )

    def prompt_user(self, model, collected):
        result = input(self.input_text(model, collected))
        return result

    def handle_input_loop(self, model):
        collected = model.get(self.field_name, [])
        done_with_this_model = False

        new_data = None

        while not done_with_this_model:
            first = True
            while first or not self.validate_input(new_data):
                if not first:
                    print('No. That value is not right!. Try again...')
                else:
                    first = False

                new_data = self.prompt_user(model, collected)

                if self.skip_input is not None and new_data == self.skip_input:
                    return False, None

                if new_data == self.done_input:
                    done_with_this_model = True
                    break

                if new_data == self.null_input:
                    collected = None
                    done_with_this_model = True
                    break

                new_data = self.clean_input(new_data)

            else:
                if self.accept_duplicates or new_data not in collected:
                    collected.append(new_data)

        return True, collected


class LoadMemory(Task):
    def __init__(self, file_path):
        super(LoadMemory, self).__init__()
        self.file_path = file_path

    def process(self, data_helper):
        return data_helper

    def setup(self, data_helper):
        memory_data = {}

        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file_object:
                memory_data = json.load(file_object)

        recursive_defaultdict = lambda: defaultdict(recursive_defaultdict)

        memory = recursive_defaultdict()
        for source in memory_data.keys():
            for field_name in memory_data[source].keys():
                for pk in memory_data[source][field_name].keys():
                    memory[source][field_name][int(pk) if pk.isdigit() else pk] = memory_data[source][field_name][pk]

        data_helper.memory = memory
        return data_helper


class SaveMemory(Task):
    def __init__(self, file_path, indent=2):
        super(SaveMemory, self).__init__()
        self.file_path = file_path
        self.indent = indent

    def process(self, data_helper):
        return data_helper

    def teardown(self, data_helper):
        with open(self.file_path, 'w') as file_object:
            json.dump(
                data_helper.memory,
                file_object,
                indent=self.indent,
                ensure_ascii=False
            )
        return data_helper


class AddIds(Task):
    source = None

    def __init__(self, source=None, initial=None):
        super(AddIds, self).__init__()
        self.source = source or self.source
        self.initial = initial or {}

    def process(self, data_helper):
        id_inc = self.initial.get(self.source, -1)

        if all(['id' not in model for model in data_helper.data[self.source]]):
            for i in range(len(data_helper.data[self.source])):
                id_inc += 1

                data_helper.data[self.source][i] = OrderedDict(
                    [('id', id_inc)] +
                    [(k, v) for k, v in data_helper.data[self.source][i].items() if k != 'id']
                )

        return data_helper


class RemoveField(Task):
    source = None

    def __init__(self, source=None, field_name=None):
        super(RemoveField, self).__init__()
        self.source = source or self.source
        self.field_name = field_name or self.field_name

    def process(self, data_helper):
        for model in data_helper.data[self.source]:
            model.pop(self.field_name, None)
        return data_helper


class RenameField(Task):
    source = None
    field_name = None
    new_name = None

    def __init__(self, source=None, field_name=None, new_name=None):
        super(RenameField, self).__init__()
        self.source = source or self.source
        self.field_name = field_name or self.field_name
        self.new_name = new_name or self.new_name

    def process(self, data_helper):
        for model in data_helper.data[self.source]:
            if self.field_name in model:
                model[self.new_name] = model.pop(self.field_name)
        return data_helper


class SortDataByAttrs(Task):
    source = None
    fields = ()

    def __init__(self, *fields, source=None):
        super(SortDataByAttrs, self).__init__()
        self.source = source or self.source
        self.fields = fields if len(fields) else self.fields

    def process(self, data_helper):
        data_helper.data[self.source] = sorted(
            data_helper.data[self.source],
            key=lambda x: tuple(x.get(y) for y in self.fields)
        )
        return data_helper


class SortData(Task):
    source = None

    @staticmethod
    def sort_function(x):
        return x

    def __init__(self, source=None, sort_function=None):
        super(SortData, self).__init__()
        self.source = source or self.source
        self.sort_function = sort_function if sort_function else self.sort_function

    def process(self, data_helper):
        data_helper.data[self.source] = sorted(
            data_helper.data[self.source],
            key=self.sort_function
        )
        return data_helper


class SortDataKeys(Task):
    source = None
    preferred_order = []

    def __init__(self, source=None, preferred_order=None):
        super(SortDataKeys, self).__init__()
        self.source = source or self.source
        self.preferred_order = preferred_order if preferred_order else self.preferred_order

    def process(self, data_helper):
        if not self.preferred_order:
            self.preferred_order = set()
            for model in data_helper.data[self.source]:
                self.preferred_order.update(model.keys())
            self.preferred_order = list(self.preferred_order)
            self.preferred_order.sort()
        else:
            for model in data_helper.data[self.source]:
                for k in model.keys():
                    if k not in self.preferred_order:
                        self.preferred_order.append(k)

        for i in range(len(data_helper.data[self.source])):
            data_helper.data[self.source][i] = OrderedDict(
                sorted(
                    data_helper.data[self.source][i].items(),
                    key=lambda x: self.preferred_order.index(x[0])
                )
            )
        return data_helper


class SortAttrData(Task):
    source = None
    attr = None

    @staticmethod
    def sort_function(x):
        return x

    def __init__(self, source=None, attr=None, sort_function=None):
        super(SortAttrData, self).__init__()
        self.source = source or self.source
        self.attr = attr or self.attr
        self.sort_function = sort_function if sort_function else self.sort_function

    def process(self, data_helper):
        for model in data_helper.data[self.source]:
            model[self.attr] = sorted(model[self.attr], key=self.sort_function)
        return data_helper


class AddHashes(Task):
    source = None
    attr = 'hash'
    exclude = []
    include = []

    def __init__(self, source=None, attr=None, exclude=None, include=None):
        super(AddHashes, self).__init__()
        self.source = source or self.source
        self.attr = attr or self.attr
        self.exclude = exclude or self.exclude
        self.include = include or self.include

    def process(self, data_helper):
        fields = []
        for model in data_helper.data[self.source]:
            for k in model.keys():
                if k not in self.exclude and (self.include == [] or k in self.include) and k not in fields:
                    fields.append(k)

        for model in data_helper.data[self.source]:
            model[self.attr] = self.get_hash(model, fields)
        return data_helper

    def get_hash(self, model, fields):
        return str(hashlib.sha512(
            "|".join([repr(model.get(a))for a in fields]).encode('utf-8')
        ).hexdigest())
