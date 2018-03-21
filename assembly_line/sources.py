import os
import errno
import sys
from collections import OrderedDict

from assembly_line.expections import AssemblyLineError


ERROR_INVALID_NAME = 123


class DataSource:

    def fetch_data(self):
        raise NotImplementedError()

    def save_data(self, data):
        raise NotImplementedError()

    def append_data(self, entry):
        raise NotImplementedError()


class FileSource:
    root = None
    path = None
    write_path = None
    source_key = None

    def __init__(self, path=None, root=None, write_path=None, source_key=None):
        self.root = root or self.root
        self.path = path or self.path
        self.write_path = write_path or self.write_path
        self.source_key = source_key or self.source_key

        if not self.write_path:
            self.write_path = self.path

        if not os.path.isfile(self.get_read_path()):
            raise AssemblyLineError(f"{self.get_read_path()} is not a file")

        if not self.is_pathname_valid(self.get_write_path()) or os.path.isdir(self.get_write_path()):
            raise AssemblyLineError(f"{self.get_write_path()} is not a valid file write path")

        if not self.source_key:
            self.source_key = '.'.join(os.path.basename(self.path).split('.')[:-1])

    def create_or_override_source(self):
        raise NotImplementedError()

    def get_read_path(self):
        return os.path.abspath(os.path.join(self.root, self.path))

    def get_write_path(self):
        return os.path.abspath(os.path.join(self.root, self.write_path))

    @staticmethod
    def is_pathname_valid(pathname):
        """
        `True` if the passed pathname is a valid pathname for the current OS;
        `False` otherwise.

        https://stackoverflow.com/questions/9532499
        """

        # If this pathname is either not a string or is but is empty, this pathname
        # is invalid.
        try:
            if not isinstance(pathname, str) or not pathname:
                return False

            # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
            # if any. Since Windows prohibits path components from containing `:`
            # characters, failing to strip this `:`-suffixed prefix would
            # erroneously invalidate all valid absolute Windows pathnames.
            _, pathname = os.path.splitdrive(pathname)

            # Directory guaranteed to exist. If the current OS is Windows, this is
            # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
            # environment variable); else, the typical root directory.
            root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
                if sys.platform == 'win32' else os.path.sep
            assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

            # Append a path separator to this directory if needed.
            root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

            # Test whether each path component split from this pathname is valid or
            # not, ignoring non-existent and non-readable path components.
            for pathname_part in pathname.split(os.path.sep):
                try:
                    os.lstat(root_dirname + pathname_part)
                # If an OS-specific exception is raised, its error code
                # indicates whether this pathname is valid or not. Unless this
                # is the case, this exception implies an ignorable kernel or
                # filesystem complaint (e.g., path not found or inaccessible).
                #
                # Only the following exceptions indicate invalid pathnames:
                #
                # * Instances of the Windows-specific "WindowsError" class
                #   defining the "winerror" attribute whose value is
                #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
                #   fine-grained and hence useful than the generic "errno"
                #   attribute. When a too-long pathname is passed, for example,
                #   "errno" is "ENOENT" (i.e., no such file or directory) rather
                #   than "ENAMETOOLONG" (i.e., file name too long).
                # * Instances of the cross-platform "OSError" class defining the
                #   generic "errno" attribute whose value is either:
                #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
                #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
                except OSError as exc:
                    if hasattr(exc, 'winerror'):
                        if exc.winerror == ERROR_INVALID_NAME:
                            return False
                    elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                        return False
        # If a "TypeError" exception was raised, it almost certainly has the
        # error message "embedded NUL character" indicating an invalid pathname.
        except TypeError:
            return False
        # If no exception was raised, all path components and hence this
        # pathname itself are valid. (Praise be to the curmudgeonly python.)
        else:
            return True


class JSONSource(DataSource):
    def process(self, data_helper):
        data = OrderedDict()
        for source_key in self.sources:
            with open(f'{self.root}{source_key}.{self.extension}', 'r') as file_object:
                data[source_key] = json.load(file_object, object_pairs_hook=OrderedDict)
            self.log.info(f'{source_key} data loaded')
        setattr(data_helper, self.attr, data)
        return data_helper