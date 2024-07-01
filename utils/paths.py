import os
from os.path import join as join_path
from os.path import isdir as is_directory


class Paths:
    """Class for managing directory paths."""
    def __init__(self):
        self.working_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_dir = self.set_root()
        os.chdir(self.root_dir)

        # Set roots
        self.Data = get_dir(join_path(self.root_dir, "Datasets"), True)
        self.Models = get_dir(join_path(self.root_dir, "Models"), True)
        self.Configs = get_dir(join_path(self.Models, "Configs"), True)
        self.Train = get_dir(join_path(self.Models, "Train"), True)
        self.Modules = get_dir(join_path(self.Models, "Modules"), True)

    def set_root(self):
        while True:
            if "DecomposeTransformer" in os.listdir(self.working_dir):
                root_path = join_path(self.working_dir, "DecomposeTransformer")
                return root_path
            parent_dir = os.path.dirname(self.working_dir)
            if parent_dir == self.working_dir:
                return None
            self.working_dir = parent_dir


def get_dir(path, flag=False):
    norm_path = os.path.normpath(path)
    segments = norm_path.split(os.sep)
    current_path = segments[0] if segments[0] else os.sep

    for segments in segments[1:]:
        current_path = join_path(current_path, segments)

        if is_directory(current_path):
            continue

        if flag:
            os.mkdir(current_path)
            return path
        else:
            return False
    if flag and not is_directory(current_path):
        os.mkdir(current_path)
    return True if not flag else path

paths = Paths()
