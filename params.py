import os
import json
from datetime import date


class Params:
    def __init__(self, config_path: str):
        if not os.path.isfile(config_path):
            raise ValueError("Path {} does not point to a file which exists".format(config_path))
        if os.path.splitext(config_path)[-1] != ".json":
            raise ValueError("Path must point to a .json file not a {} file.".format(os.path.splitext(config_path)[-1]))
        self.update(config_path)

    def save(self, json_path=None):
        """Saves parameters to json file"""
        if json_path is None:
            json_path = os.path.join(self.directory, "params.json")
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            # params["directory"] = make_unique_directory(
            #     "experiments/" + params["model"] + "_" + params["spektral_datasets"] + "_" + format_date() + "_{}")
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def format_date():
    return date.isoformat(date.today()).replace("-", "_")
