from pathlib import Path
from functools import lru_cache
import os
import re
import pandas as pd
#########################
### --- Constants --- ###
#########################

## Colours ##
COL_BLUE = '\033[94m'  # Light blue
COL_DEFAULT = '\033[0m'  # Reset to default color

## Type Dicts ##
general_type_dict = {"log": {
                            "file": str,
                            "console": str
                            }
                    }

#########################
### --- Helpers --- ###
#########################

def find_project_root(marker_file_name: str = "pyproject.toml") -> Path:
    """
    Traverses up the directory tree from the current script's location
    to find the project root, identified by the presence of a marker file.

    Args:
        marker_file_name: The name of the file that marks the project root
                          (e.g., 'project.toml', 'pyproject.toml', '.git').

    Returns:
        A pathlib.Path object representing the project root directory.

    Raises:
        FileNotFoundError: If the marker file cannot be found by traversing up.
    """
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if (parent / marker_file_name).exists():
            return parent
    raise FileNotFoundError(f"Project root marker '{marker_file_name}' not found "
                            f"in {current_dir} or any parent directories.")

def type_check(dict_to_check, type_dict):
   # TODO: add auto conversion (i.e from int to float where possible)

    for (key, value) in type_dict.items():
        if isinstance(type_dict[key], type): # check for type instanced 
            if not isinstance(dict_to_check[key], value):
                return f"Expected type {value} for {key}, got {type(dict_to_check[key])}"
        elif isinstance(type_dict[key], tuple): # check for array-like (0) filled with type (1)
            if not isinstance(dict_to_check[key], type_dict[key][0]) and all(isinstance(item,  type_dict[key][1]) for item in dict_to_check[key]):
                return f"Expected type {type_dict[key][0]} filled with {type_dict[key][1]} for {key}, got {type(dict_to_check[key])} filled {[type(x) for x in dict_to_check[key]]}"
        elif isinstance(type_dict[key], dict): # recursively process nested dictionaries 
            type_check(dict_to_check[key], type_dict[key])
        else:
            raise ValueError(f"Unexpected type contained in type dict:  {type_dict}")
        
    return False

def dict_key_check(dict_to_check, dict_to_compare):
    for (key, value) in dict_to_compare.items():
        if not isinstance(value, dict):
            if key not in dict_to_check.keys():
                return key
        else:
            dict_key_check(dict_to_check[key], value)
    return False

def key_check(step_config, type_dict):
    missing_in_config = dict_key_check(step_config, type_dict) # check that all type dict keys are in config keys 
    missing_in_type = dict_key_check(type_dict, step_config) # check that all config keys are in the type dict

    if missing_in_config:
        return f"Expected {missing_in_config} to be in config keys. {COL_BLUE}Check config.toml{COL_DEFAULT}"
    if missing_in_type:
        return f"Expected {missing_in_type} to be in type_dict keys. {COL_BLUE}Update type_dict in config.py{COL_DEFAULT}"

def config_init_check(config, type_dict, step_name):
    key_error = key_check(config, type_dict)
    if key_error:
        raise ValueError(f"{step_name} | {key_error}")
    type_error = type_check(config, type_dict)
    if type_error:
        raise TypeError(f"{step_name} | {type_error}")
    
############################
### --- Load configs --- ###
############################

@lru_cache()
def _load_config(path: Path = None) -> dict:
    """
    Loads and caches the TOML configuration from the specified path.
    """
    if path is None:
        path = find_project_root() / "config" / "config.toml"
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import toml
        return toml.load(path)
    except ImportError:
        import tomli
        with open(path, "rb") as f:
            return tomli.load(f)

def get_data_config():
    cfg = _load_config()["data"]
    for path in ["database_loc"]:
        cfg[path] = Path(cfg[path])
    return cfg

def get_model_config():
    return _load_config()["model"]

def get_train_config():
    config = _load_config()["train"]
    if config["test_size"] > config["CV_delta"]:
        raise ValueError("Test size cannot be larger than CV delta, TODO: change to warning")
    return config 
 
def get_log_config():
    return _load_config()["log"]
