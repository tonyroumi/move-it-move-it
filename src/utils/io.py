import json
import yaml

class DataUtils:
    """ Lightweight data-loading utilities. """

    @staticmethod
    def load_yaml(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)