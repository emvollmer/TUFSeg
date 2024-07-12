#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for training U'Net model from segmentation-models toolbox

Author: Leon Klug, Elena Vollmer
Date: 26.06.2023
"""
# import built-in dependencies
import ast
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path, PosixPath
import shutil
import sys
import time
import warnings

# set XLA_FLAGS environment variable for TensorFlow
cuda_data_dir = os.path.join(os.environ['CUDA_HOME'], 'nvvm/libdevice')
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_data_dir}'

# suppress messages from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing (type, 1) or '1type' as a synonym of type is deprecated")

# import external dependencies
import keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from keras.layers import Input, Conv2D

# import module dependencies
from scripts.configuration import init_temp_conf, update_conf, cp_conf, _default_config
config = init_temp_conf()
from scripts.segm_models._utils import (
    configure_logging, ImageProcessor, MaskProcessor
)
# from scripts.segm_models import utils

# Very important! Enables Usage of segmentation models library
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

logging.getLogger('h5py').setLevel(logging.ERROR)
# --------------------------------------
_logger = logging.getLogger("train")
log_file = Path()

NUM_CLASSES = len(config['data']['masks']['labels']) + 1    # to account for the background class


class GetKeyValuePairs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            # use int if value is a digit, otherwise stay with string
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = value


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays in model config dumping
    by turning them into lists
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('-dst', '--dst-model-dir', dest='model_root', required=True,
                        help='Path to destination directory to save model to.')
    parser.add_argument('-src', '--src-data-dir', dest='data_root',
                        help='(Optional) Path to source directory of all data. '
                             'If not provided, value from temp config (defined during setup) '
                             'will be used.')
    parser.add_argument('-ch', '--channels', dest='channels', type=int, default=4,
                        choices=[4, 3], help='Whether to process the data as 3 channels '
                                             '(greyRGB+T+T) or keep the 4 channels (RGBT).')
    parser.add_argument('-proc', '--processing', dest='processing', type=str, default="basic",
                        choices=["basic", "vignetting", "retinex_unsharp"],
                        help='Whether to apply additional filters to the data (retinex unsharp, '
                             'vignetting removal) or keep as basic (RGBT).')
    parser.add_argument('--only-tir', dest='only_tir', action='store_true',
                        help='Whether to use only thermal image (not RGB). If the flag is added, '
                             'the inputs will automatically be 3 channels and the selected '
                             'processing option is applied.')
    parser.add_argument('-cfg', '--cfg-options', dest='cfg_options', nargs='+',
                        action=GetKeyValuePairs,
                        help='Specify training cfg options in the following manner: '
                             '"seed=1000 lr=0.001 epochs=35 batch_size=8". Without these, the '
                             'listed examples will be chosen as defaults')

    # Combine log level options into one with a default value
    log_levels_group = parser.add_mutually_exclusive_group()
    log_levels_group.add_argument('--quiet', dest='log_level', action='store_const',
                                  const=logging.WARNING, help='Show only warnings.')
    log_levels_group.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                                  const=logging.INFO, help='Show verbose log messages (info).')
    log_levels_group.add_argument('-vv', '--very-verbose', dest='log_level', action='store_const',
                                  const=logging.DEBUG, help='Show detailed log messages (debug).')
    log_levels_group.set_defaults(log_level=logging.WARNING)
    return parser.parse_args()


def main(
        model_root: Path,
        data_root: Path = None,
#        split_root: Path = None,
        channels: int = 4,
        processing: str = "basic",
        only_tir: bool = False,
        cfg_options: dict = None,
        log_level=logging.WARNING
):
    """
    Coordinate model training with user inputs
    """
    # create model timestamp folder
    model_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = Path(model_root, model_folder)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    # set up logging with log file in model directory
    global log_file
    log_file = Path(model_dir, "log.log")
    configure_logging(_logger, log_level, log_file)
    _logger.info('---TRAINING LOG---')

    # check for only_tir flag, which requires 3 channel inputs
    if channels != 3 and only_tir == True:
        _logger.info(f'Flag "only_tir" has been set, so only 3 channel inputs will be used. '
                     f'Changing currently defined channel value of {channels} to 3...')
        channels = 3

    # log local function arguments, including the user provided ones
    main_args = locals()
    if main_args.get('cfg_options', {}).get('encoded_weights') in ["None", "none"]:
        main_args['cfg_options']['encoded_weights'] = None

    for k, v in main_args.items():
        _logger.info(f'Parsed user argument - {k}:\t{v}')

    # update config file with those local arguments that aren't = None
    update_conf(conf=config, params={k: v for k, v in main_args.items() if v is not None})

    # load training and testing data
    _logger.info(f"Load data into {channels} channel images and process with "
                 f"'{processing}' option.")
    X_train, y_train, X_test, y_test = load_data()

    # onehot encode y data (masks)
    y_train_onehot = np.asarray(tf.one_hot(y_train, NUM_CLASSES))
    y_test_onehot = np.asarray(tf.one_hot(y_test, NUM_CLASSES))

    # train
    _logger.info(f"Model cfg options:\n{config['model']}")
    model = train(
        X_train, y_train_onehot, X_test, y_test_onehot,
        model_path=Path(model_dir, config['model']['type'] + config['model']['suffix'])
    )
    save_model_info(model=model, model_dir=model_dir)

    # save json config to model directory
    cp_conf(model_dir)
    _logger.info(f"Saved configuration of training run to {model_dir}")


def load_data():
    """
    Load and process data according to filters in utils.py

    :return: X_train, y_train, X_test, y_test
    """
    start = time.time()
    img_proc = ImageProcessor(config)
    X_train = img_proc.process_images(root="train")
    X_test = img_proc.process_images(root="test")

    mask_proc = MaskProcessor(config)
    y_train = mask_proc.load_masks(root="train")
    y_test = mask_proc.load_masks(root="test")

    duration = time.time() - start
    _logger.info(f"Elapsed time during data loading:\t{round(duration / 60, 2)} min")
    return X_train, y_train, X_test, y_test


class CustomEpochLogger(tf.keras.callbacks.Callback):
    """
    Custom training logger to save epoch information to logging file
    """
    def __init__(self):
        super().__init__()
        self.current_batch = 0  # Initialize the batch number

    def on_epoch_begin(self, epoch, logs=None):
        self.current_batch = 0
        self.epoch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        self.current_batch += 1

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        log_message = f'\n--- Epoch {epoch + 1}/{self.params["epochs"]} ---\n'
                      
        log_message += f'Batch: {self.current_batch}/{self.params["steps"]} --- ' \
                       f'Time: {round(epoch_duration)}s/epoch - ' \
                       f'{round((epoch_duration / self.params["steps"]) * 1000)}ms/step\n'

        train_logs = {k: v for k, v in logs.items() if "val_" not in k}
        log_message += f'Training: {" - ".join([f"{k}: {round(v, 4)}" for k, v in train_logs.items()])}\n'

        val_logs = {k: v for k, v in logs.items() if "val_" in k}
        log_message += f'Validation: {" - ".join([f"{k}: {round(v, 4)}" for k, v in val_logs.items()])}\n'

        # Log to file
        with open(log_file, "a") as f:
            f.write(log_message)


def train(X_train, y_train_onehot, X_test, y_test_onehot, model_path: Path):
    """
    Train UNet model with training images (X_train) and masks (y_train)
    and user-defined parameters.
    Use test dataset only to visualise performance before final evaluation.
    X data can be 3 channel or of different dimension (UNet will automatically be adapted).

    :param X_train: train images
    :param y_train_onehot: corresponding onehot encoded train masks
    :param X_test: test images
    :param y_test_onehot: corresponding onehot encoded test masks
    :param model_path: (Path) path to which model will be saved
    :return: model saved to provided path
    """
    start = time.time()
    # Check that provided path can be used for saving model
    assert type(model_path) == PosixPath and model_path.suffix == ".hdf5", \
        f"Provided model path '{model_path}' is not a Path ending in '.hdf5'!"

    cfg = config['train']
    _logger.info(f"Training cfg options:\n{cfg}")

    # Define number of channels, while influences training
    N = X_train.shape[-1]

    # Define the model
    _logger.info("Loading model backbone...")

    SIZE_H = config['data']['loader']['SIZE_H']
    SIZE_W = config['data']['loader']['SIZE_W']

    if N == 3:
        model = sm.Unet(config['model']['backbone'],
                        encoder_weights=config['model']['encoded_weights'],
                        classes=NUM_CLASSES, input_shape=(SIZE_H, SIZE_W, N))
    else:
        _logger.info(f"Channel count of {N} != 3. Adapting UNet by including a fitting first layer...")
        base_model = sm.Unet(backbone_name=config['model']['backbone'],
                             encoder_weights=config['model']['encoded_weights'],
                             classes=NUM_CLASSES)

        inp = Input(shape=(SIZE_H, SIZE_W, N))
        layer_1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
        out = base_model(layer_1)

        model = keras.models.Model(inputs=inp, outputs=out, name=base_model.name)

    # Compile the model with loss function and optimizer loaded from config
    loss_name = cfg['loss']['name']
    try:
        loss_function = getattr(tfa.losses, loss_name)
    except AttributeError:
        raise ValueError(f"Loss function '{loss_name}' not found in tensorflow_addons.losses!")

    optimizer_name = cfg['optimizer']['name']
    try:
        optimizer = getattr(tf.keras.optimizers, optimizer_name)
    except AttributeError:
        raise ValueError(f"Optimizer {optimizer_name} not found in tensorflow.keras.optimizers!")

    _logger.info(f"Compiling model with '{loss_name}' loss function and '{optimizer_name}' optimizer.")
    model.compile(
        optimizer=optimizer(learning_rate=cfg['optimizer']['lr']),
        loss=loss_function(alpha=cfg['loss']['alpha'], gamma=cfg['loss']['gamma']),
        metrics=[eval(v) for v in config['eval']['SM_METRICS'].values()]
    )

    tf.random.set_seed(cfg['seed'])

    # Log model history to a CSV file
    CSVHistoryLogger = keras.callbacks.CSVLogger(Path(model_path.parent, 'training_history.csv'),
                                                 separator=',', append=False)

    # Fit the Model
    _logger.info("Training model...")
    model.fit(X_train, y_train_onehot,
              batch_size=cfg['batch_size'],
              verbose=2,
              epochs=cfg['epochs'],
              validation_data=(X_test, y_test_onehot),
              callbacks=[CustomEpochLogger(), CSVHistoryLogger]
    )

    duration = time.time() - start
    _logger.info(f"Elapsed time during model training:\t{round(duration / 60, 2)} min")

    # Save trained Model
    model.save(model_path)
    _logger.info(f"Model saved to '{model_path}'.")

    return model


def save_model_info(model, model_dir: Path):
    """
    Save relevant model information to model folder

    :param model: trained tensorflow model
    :param model_dir: (Path) directory to model outputs
    """
    # specify paths
    summary_path = Path(model_dir, 'model_summary.txt')
    config_path = Path(model_dir, 'model_config.json')

    # write to the files
    with open(summary_path, 'w') as summary_file:
        sys.stdout = summary_file  # Redirect print statements to the file
        model.summary()      # Print the summary to the file

    _logger.info(f"Model summary saved to '{summary_path}'")

    with open(config_path, 'w') as json_file:
        json.dump(model.get_config(), json_file, 
                  cls=CustomJSONEncoder, indent=4)  # encoder for non-serializable objects

    _logger.info(f"Model configuration saved to '{config_path}'")


if __name__ == "__main__":
    args = parse_args()
    main(
        model_root=args.model_root,
        data_root=args.data_root,
        channels=args.channels,
        processing=args.processing,
        only_tir=args.only_tir,
        cfg_options=args.cfg_options,
        log_level=args.log_level
    )
