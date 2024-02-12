import os


class Paths:
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_dir = self.set_root()
        self.get_root()

        # Set roots
        self.Data = self.get_dir(
            self.root_dir, "Datasets"
        )  # path of the training dataset
        self.Model = self.get_dir(self.root_dir, "Models")
        self.Train = self.get_dir(
            self.Model, "Train"
        )  # path of the trained models
        self.Config = self.get_dir(
            self.Model, "Config"
        )  # path of the model configuration files

    def set_root(self):
        while True:
            if "DecomposeBERT" in os.listdir(self.cur_dir):
                root_path = os.path.join(self.cur_dir, "DecomposeBERT")
                return root_path
            par_dir = os.path.dirname(self.cur_dir)
            if par_dir == self.cur_dir:
                return None
            self.cur_dir = par_dir

    def get_root(self):
        os.chdir(self.root_dir)
        return self.root_dir

    @staticmethod
    def is_dir(_dirname):
        return os.path.isdir(_dirname)

    @staticmethod
    def is_file(_filename):
        return os.path.isfile(_filename)

    def get_dir(self, _dirname1, _dirname2):
        _path = os.path.join(_dirname1, _dirname2)
        if not self.is_dir(_path):
            os.mkdir(_path)
        return _path


p = Paths()
