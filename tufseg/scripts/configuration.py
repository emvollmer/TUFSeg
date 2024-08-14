import json
from pathlib import Path, PosixPath
import shutil

# define temporary file directory
temp_conf_path = Path(Path.home(), ".config/segm_model.json")

_default_config = {
    "data_root": "/path/to/data/",
    "anno_root": "/path/to/data/annotations/jsons/",
    "split_root": "/path/to/split/",
    "img_folder": "images",
    "mask_folder": "masks",
    "model_root": "/path/to/models/",
    "model_folder": "timestamp/",
    "split": ["train", "test"],
    "data": {
        "ANNO_IMG_SHAPE": [512, 640],
        "loader": {
            "SIZE_H": 512,  # alternatively: 256
            "SIZE_W": 640,  # alternatively: 320
            "channels": 4,
            "processing": "basic",
            "only_tir": False,
        },
        "annotations": {
            "descriptor": "thermal_objects",
            "classes": ["building", "car (cold)", "car (warm)", "manhole (round) cold",
                        "manhole (round) warm", "manhole (square) cold", "manhole (square) warm",
                        "miscellaneous", "person", "street lamp cold", "street lamp warm"]
        },
        "masks": {
            "labels": ["building", "car (cold)", "car (warm)", "manhole (round)",
                       "manhole (square)", "miscellaneous", "person", "street lamp"],
            "custom_colors": [[0.16, 0.16, 0.16], "brown", "blue", "yellowgreen", "yellow",
                              "magenta", "red", "orange", "lightgray", "aqua", "white", "coral",
                              "teal", "pink", "goldenrod", "green", "plum", "purple", "silver",
                              "indigo"]
        }
    },
    "model": {
        "type": "UNet",
        "backbone": "resnet152",
        "encoded_weights": "imagenet",
        "suffix": ".hdf5"
    },
    "train": {
        "epochs": 35,
        "batch_size": 8,
        "seed": 1000,
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
        },
        "loss": {
            "name": "SigmoidFocalCrossEntropy",
            "function": "tfa.losses.SigmoidFocalCrossEntropy",
            "alpha": 0.3,
            "gamma": 3
        },
    },
    "eval": {
        "SKLEARN_METRICS": {
            "precision": {
                "func": "precision_score",
                "params": {"average": "macro", "zero_division": 0.0}
            },
            "weighted_precision": {
                "func": "precision_score",
                "params": {"average": "weighted", "zero_division": 0.0}
            },
            "accuracy": {
                "func": "accuracy_score",
                "params": {}
            },
            "balanced_accuracy": {
                "func": "balanced_accuracy_score",
                "params": {}
            },
            "f1": {
                "func": "f1_score",
                "params": {"average": "weighted"}
            },
            "iou": {
                "func": "jaccard_score",
                "params": {"average": "macro"}
            },
            "weighted_iou": {
                "func": "jaccard_score",
                "params": {"average": "weighted"}
            }
        },
        "SM_METRICS": {
            "iou_score": "sm.metrics.iou_score",
            "f1_score": "sm.metrics.f1_score",
            "precision": "sm.metrics.precision",
            "recall": "sm.metrics.recall"
        }
    }
}


def init_temp_conf(delete_existing: bool = False):
    """
    Create a temporary config file according to _default_config.
    Check if temporary file has been created and load config from there if that is the case

    :param delete_existing: (bool) If True, creates new temp config, even if one already exists
    :returns: config - dictionary of configuration settings
    """
    if temp_conf_path.is_file() and delete_existing is False:
        try:
            config = read_conf(temp_conf_path)
        except json.JSONDecodeError:
            print("Previously saved config is corrupted or empty. Deleting and recreating...")
            temp_conf_path.unlink()

            config = _default_config
            write_conf(config, temp_conf_path)
    else:
        config = _default_config
        write_conf(config, temp_conf_path)

    return config


def write_conf(conf: dict, config_path: Path):
    """
    Write config to json file path.

    :param conf: dictionary of configuration settings
    :param config_path: Path to temporary config file
    :raises TypeError: If config_path not a .json
    """
    if config_path.suffix not in ['.json']:
        raise TypeError(f"Only .json configs allowed, but config is of type {config_path.suffix}!")

    with open(config_path, "w") as f:
        json.dump(conf, f, indent=4)


def read_conf(config_path: Path) -> dict:
    """
    Read config from json file path.

    :param config_path: Path to temporary config file
    :raises TypeError: If config_path not a .json
    :raises FileNotFoundError: If config_path does not exist / isn't a file
    :returns: conf - dictionary of configuration settings
    """
    if config_path.suffix not in ['.json']:
        raise TypeError(f"Only .json configs allowed, but config is of type {config_path.suffix}!")

    if not config_path.is_file():
        raise FileNotFoundError(f"No configuration file found at '{config_path}'")

    with open(config_path, "r") as f:
        conf = json.load(f)

    return conf


def update_conf(conf: dict, params: dict):
    """
    Update config by replacing values of provided parameters
    in dictionary form

    :param conf: dictionary of configuration values
    :param params: dictionary of values to update
    """
    # flatten params dictionary if it's nested
    if any(isinstance(v, dict) for v in params.values()):
        params = flatten_dict(params)

    # replace PosixPath instances with strings
    for k, v in params.items():
        if isinstance(v, PosixPath):
            params[k] = str(v)

    # update configuration dictionary
    for key, value in conf.items():
        if isinstance(value, dict):
            update_conf(value, params)
        else:
            if key in params.keys():
                conf[key] = params[key]

    # update temporary config file
    write_conf(conf, temp_conf_path)


def flatten_dict(d: dict) -> dict:
    """
    Flatten a nested dictionary, keeping only the innermost keys and values.
    If the same key exists in multiple nestings, the value from the innermost
    key will overwrite all others.

    :param d: Nested dictionary to be flattened
    :return: Flattened dictionary containing innermost keys and values
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


def cp_conf(dst_dir: Path):
    """
    Copy temporary config file to destination folder.
    Revert the temporary file back to the default values except for directories.

    :raises OSError: If file can't be copied
    :param dst_dir: Path of folder to which to copy config
    """
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)

    try:
        shutil.copy(temp_conf_path, Path(dst_dir, "run_config.json"))
    except OSError as e:
        print(f"The temporary configuration file could not be moved to "
              f"the destination folder '{dst_dir}'.\nAn error occurred: {e}")

    # Revert remaining temporary file to the default values, but keep the directories
    temp_config = read_conf(temp_conf_path)

    default_values = ['data', 'model', 'train', 'eval', 'model_root', 'model_folder']
    for v in default_values:
        temp_config[v] = _default_config[v]

    write_conf(temp_config, temp_conf_path)


def mv_conf(dst_dir: Path):
    """
    Move temporary config file to destination folder

    :raises OSError: If file can't be moved
    :param dst_dir: Path of folder to which to copy config
    """
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)

    try:
        shutil.move(temp_conf_path, Path(dst_dir, "run_config.json"))
    except OSError as e:
        print(f"The temporary configuration file could not be moved to "
              f"the destination folder '{dst_dir}'.\nAn error occurred: {e}")
