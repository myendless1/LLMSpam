import yaml

def convert_scientific_to_float(x):
    if isinstance(x, dict) or isinstance(x, list) or \
    isinstance(x, tuple) or isinstance(x, float) or \
    isinstance(x, int) or x is None:
        return x
    if ("e" in x or "E" in x or "e-" in x or "E-" in x) \
    and x.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).replace("E", "", 1).isdigit():
        x = float(x)
    return x

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, convert_scientific_to_float(value))

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def dict(self):
        return self.__dict__

def get_config(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
        return Config(config_dict)