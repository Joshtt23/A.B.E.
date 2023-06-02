import os
from importlib.machinery import SourceFileLoader

class ConfigFile:
    def __init__(self):
        self.config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.py"))

    def update_config(self, config):
        config_module = SourceFileLoader("config", self.config_file_path).load_module()
        config_class = getattr(config_module, "Config")
        config_dict = vars(config_class)

        for key, value in config_dict.items():
            if key != "__builtins__":
                setattr(config, key, value)

    def save_config(self, config):
        with open(self.config_file_path, "w") as file:
            file.write("class Config:\n")
            for key, value in vars(config).items():
                if key != "__builtins__":
                    if isinstance(value, str):
                        file.write(f'    {key} = "{value}"\n')
                    else:
                        file.write(f"    {key} = {value}\n")
