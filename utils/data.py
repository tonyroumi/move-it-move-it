import json
import yaml


class DataUtils:
    """ Data-loading utilities. """

    @staticmethod
    def load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
