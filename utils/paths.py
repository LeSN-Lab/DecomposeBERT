import os


class Paths:
    def __init__(self):
        self.cur_dir = os.getcwd()
        self.root_dir = self.set_root()
        self.get_root()

        # Set roots
        self.Data = os.path.join(self.root_dir, "Datasets")  # path of the training dataset
        self.Model = os.path.join(self.root_dir, "Models")
        self.train_path = os.path.join(
            self.Model, "Train"
        )  # path of the trained models
        self.model_config = os.path.join(
            self.Model, "Config"
        )  # path of the model configuration files

        self.check_dir(self.Data)
        self.check_dir(self.Model)
        self.check_dir(self.train_path)
        self.check_dir(self.model_config)

        # Specific directories
        self.model_dir = None
        self.train_dir = None
        self.data_dir = None

        # others
        self.model_name = None

    def check_dir(self, dir):
        flag = True
        if not os.path.isdir(dir):
            flag = False
            os.mkdir(dir)
        return flag

    def set_root(self):
        while True:
            if "DecomposeBERT" in os.listdir(self.cur_dir):
                root_path = os.path.join(self.cur_dir, "DecomposeBERT")
                return root_path
            par_dir = os.path.dirname(self.cur_dir)
            if par_dir == self.cur_dir:
                return None
            self.cur_dir = par_dir

    # Set path of the current model
    def set_model_path(self, dirname):
        self.model_dir = os.path.join(self.model_config, dirname)

    def set_train_path(self, dirname):
        self.train_dir = os.path.join(self.train_path, dirname)

    def set_data_dir(self, dirname):
        self.data_dir = os.path.join(self.Data, dirname)

    def set_model_name(self, name):
        self.model_name = name

    def set(self, model_path, model_name):
        self.set_model_path(model_path)
        self.set_train_path(model_path)
        self.set_data_dir(model_path)
        self.set_model_name(model_name)

    def get_root(self):
        os.chdir(self.root_dir)
        return self.root_dir

    # Get path of the current model
    def get_model_path(self):
        return self.model_dir

    def get_train_path(self):
        return self.train_path

    def get_data_dir(self):
        return self.data_dir


p = Paths()
