import os
from os.path import join as join
from os.path import isdir as isdir


class Paths:
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_dir = self.set_root()
        os.chdir(self.root_dir)

        # Set roots
        self.Data = get_dir(join(self.root_dir, "Datasets"), True)
        self.Models = get_dir(join(self.root_dir, "Models"), True)
        self.Configs = get_dir(join(self.Models, "Configs"), True)
        self.Train = get_dir(join(self.Models, "Train"), True)
        self.Modules = get_dir(join(self.Models, "Modules"), True)

    def set_root(self):
        while True:
            if "DecomposeBERT" in os.listdir(self.cur_dir):
                root_path = join(self.cur_dir, "DecomposeBERT")
                return root_path
            par_dir = os.path.dirname(self.cur_dir)
            if par_dir == self.cur_dir:
                return None
            self.cur_dir = par_dir


def get_dir(path, flag=False):
    segments = path.split(os.sep)
    current_path = segments[0] if segments[0] else os.sep

    for segments in segments[1:]:
        current_path = os.path.join(current_path, segments)

        if isdir(current_path):
            continue

        if flag:
            os.mkdir(current_path)
            return path
        else:
            return False

    return True if not flag else path


p = Paths()
