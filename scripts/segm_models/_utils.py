#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper functions and classes to train, evaluate and infer using segmentation_models toolbox
"""
# import built-in dependencies
import logging
import os
from pathlib import Path, PosixPath
import sys
from typing import Union
import warnings

# suppress messages from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing (type, 1) or '1type' as a synonym of type is deprecated")

# import external dependencies
import cv2
import keras
from keras.models import load_model
import numpy as np
from skimage.filters import unsharp_mask
from skimage.transform import resize
import tensorflow_addons as tfa     # don't remove, called during eval of config value
from tqdm import tqdm

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm    # don't remove, called during eval of config value
# --------------------------------------
logging.getLogger('h5py').setLevel(logging.ERROR)
_logger = logging.getLogger(__name__)


def configure_logging(logger, log_level: int, log_file=None):
    """Define basic logging configuration

    :param logger: logger
    :param log_level: User defined input
    :param log_file: If not None, save logging information to provided file path
    """
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # check for old handlers and remove if existing (prevents duplicate printing)
    if logger.handlers:
        _logger.warning(f"--- _utils --- old handlers will be removed:\n{logger.handlers}")
    logger.handlers = []

    # Define logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    if log_file:
        # Define logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        # set log level of logger itself, as permissive as the most permissive of its handlers
        logger.setLevel(logging.DEBUG if log_level == 10 else logging.INFO)
    else:
        # set log level of logger itself, as permissive as the most permissive of its handlers
        logger.setLevel(log_level)


def read_txt_file(txt_path: Path):
    """
    Read lines from text file and return list

    :param txt_path: (Path) textfile path (f.e. ".../train.txt") containing paths "/dataset/img.npy"
    :return: pth_list - list of Paths of image identifiers
    """
    with open(txt_path, "r") as f:
        data_list = f.read().splitlines()

    pth_list = [Path(line) for line in data_list]

    return pth_list


class Base:
    """
    Basic definitions and methods to be inherited by other classes
    """
    def __init__(self, config):
        self.config = config
        self.SIZE_H = self.config['data']['loader']['SIZE_H']
        self.SIZE_W = self.config['data']['loader']['SIZE_W']
        self.CHANNELS = self.config['data']['loader']['channels']
        self.NUM_CLASSES = len(self.config['data']['masks']['labels']) + 1  # account for background
        self.valid_roots = self.config['split']

        self.data_root = Path(self.config['data_root'])
        self.split_root = Path(self.config['split_root'])

    def save_npy(self, npy_arr: np.array, dst_dir: Path, npy_id: Path):
        """
        Save numpy to a Path "dst_dir/npy_id_parent/npy_id_name"

        :param npy_arr: array to save
        :param dst_dir: destination directory
        :param npy_id: numpy identifier (f.e. .../KA_01/DJI_...npy)
        """
        img_path = Path(dst_dir, npy_id.parent.name, npy_id.name)
        np.save(str(img_path), npy_arr)

    def check_file_exists(self, file_path: Path):
        """Check to ensure a file exists at the given path"""
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"File doesn't exist at provided path '{file_path}'!")

    def check_folder_exists(self, folder_path: Path):
        """Check to ensure a file exists at the given path"""
        if not Path(folder_path).is_dir():
            raise FileNotFoundError(f"Folder doesn't exist at provided path '{folder_path}'!")

    def define_path(self, path: Union[str, Path]):
        """Define a provided path as either Path or None, depending on if it exists."""
        if type(path) in [str, PosixPath] and Path(path).exists():
            return Path(path)
        else:
            return None


class ImageProcessor(Base):
    """
    Image processor
    """
    def __init__(self, config):
        super().__init__(config)
        self.src_img_dir = Path(self.data_root, config['img_folder'])

        self.processing = config['data']['loader']['processing']
        self.only_tir = config['data']['loader']['only_tir']

        # make sure class method exists to handle defined processing option
        proc_method_name = f"_process_{self.processing}"
        self.proc_method = getattr(self, proc_method_name, None)
        if self.proc_method is None:
            raise ValueError(f"Invalid image processing option {self.processing}!")

    def process_images(self, root: str, src_img_dir: Path = None, overwrite=True):
        """
        Load and preprocess images, so they're ready to use as input for the model

        :param src_img_dir: (Path) source directory for images before dataset split
                - f.e. "/.../data/images/"
        :param root: (str) String denoting dataset - f.e. "train", "test"
        :param overwrite: (bool) if True, force overwrite of files regardless of existence
        :raises ValueError: If "root" not one of "valid_roots"
        :return: processed images (array of mult-channel images for model inputs)
        """
        if root not in self.valid_roots:
            raise ValueError(f"Invalid root value. It must be one of '{self.valid_roots}'.")

        # define source directory - user input overrides config definition
        src_img_dir = Path(src_img_dir or self.src_img_dir)
        self.check_folder_exists(src_img_dir)

        # check if split directory exists and, if so, what images it contains
        split_img_dir = Path(self.split_root, root, self.config['img_folder'])
        if split_img_dir.is_dir():
            if overwrite is True:
                save = True
                print(f"Saving images to '{split_img_dir}'...")
            else:
                save = False
                print(f"Loading images from previously saved at '{split_img_dir}'...")
            saved_images = sorted(list(split_img_dir.glob("**/*.npy")))
        else:
            save = False
            print(f"Loading images from '{src_img_dir}', not saving them...")
            split_img_dir = src_img_dir
            saved_images = []

        # get IDs of images to be processed
        image_IDs = read_txt_file(Path(split_img_dir.parent, root + ".txt"))

        # process images - if not yet previously done, process; otherwise load from path
        proc_images = []
        if not saved_images or len(image_IDs) != len(saved_images) or overwrite:
            _logger.info(f"Preprocessing images according to '{self.processing}' method...")

            for img_path in tqdm(sorted([Path(src_img_dir, m) for m in image_IDs])):
                proc_img = self.proc_method(img_path)
                if save:
                    self.save_npy(npy_arr=proc_img, dst_dir=split_img_dir, npy_id=img_path)
                proc_images.append(proc_img)
        else:
            _logger.info(f"Loading previously preprocessed images from '{split_img_dir}'...")
            for img_path in tqdm(saved_images):
                # load image from model train test split path
                proc_img = np.load(str(img_path))
                proc_images.append(proc_img)

        return np.array(proc_images)

    def load_image(self, img_path):
        """
        Load and resize image

        :param img_path: Path to numpy image
        :raises AssertionError: If aspect ratio of config defined image width and height doesn't
        match loaded images
        :return img
        """
        # Load image from source path
        self.check_file_exists(img_path)
        img = np.load(str(img_path))

        # Resize image to size of choice (of the same aspect ratio)
        height, width, dim = img.shape
        assert width / height == self.SIZE_W / self.SIZE_H, \
            (f"Aspect ratios of image {(width, height)} and "
             f"dst shape {(self.SIZE_W, self.SIZE_H)} don't match!")

        return np.asarray(cv2.resize(img, (self.SIZE_W, self.SIZE_H)))

    def _process_basic(self, img_path):
        """
        Prepare the image data to be inputted as X data into the model: no filters

        - 4 channels: R + G + B + T
        - 3 channels: grayRGB + T + T

        If only_tir flag is set to True:

        - 3 channels: T + T + T

        :param img_path: Path to 4 channel image array (R, G, B, T)
        :raises ValueError: If channel count not 3 or 4
        :return: preproc_img as a 3 or 4 channel image array for model
        """
        img = self.load_image(img_path)

        # process RGB
        rgb_img = img[:, :, :3]
        # process Thermal
        tir_img = img[:, :, 3]

        # --- if only_tir flag is True, use only T channel
        if self.only_tir == True:
            preproc_img = np.stack((tir_img, tir_img, tir_img), axis=2)
        else:
            preproc_img = self._stack_array(rgb_img, tir_img)

        return preproc_img

    def _process_vignetting(self, img_path):
        """
        Prepare the image data to be inputted as X data into the model: vignetting removal

        - 4 channels: R + G + B + T_no_vignetting
        - 3 channels: grayRGB + T_no_vignetting + T_no_vignetting

        If only_tir flag is set to True:

        - 3 channels: T_no_vignetting + T_no_vignetting + T_no_vignetting

        :param img_path: Path to 4 channel image array (R, G, B, T)
        :raises ValueError: If channel count not 3 or 4
        :return: preproc_img as a 3 or 4 channel image array for model
        """
        img = self.load_image(img_path)

        # process RGB
        rgb_img = img[:, :, :3]

        # process Thermal
        tir_img = img[:, :, 3]
        # apply correction to thermal to remove vignetting
        tir_img = self._remove_vignetting(tir_img)

        # --- if only_tir flag is True, use only T channel
        if self.only_tir == True:
            preproc_img = np.stack((tir_img, tir_img, tir_img), axis=2)
        else:
            preproc_img = self._stack_array(rgb_img, tir_img)

        return preproc_img

    def _process_retinex_unsharp(self, img_path):
        """
        Prepare the image data to be inputted as X data into the model: retinex parvo filter, unsharp mask

        - 4 channels: grayRGB_retinex + T_retinex + grayRGB_um + T_um
        - 3 channels: grayRGB_retinex + T_retinex + T_um
        
        If only_tir flag is set to True:

        - 3 channels: T_no_vignetting + T_retinex + T_um

        :param img_path: Path to 4 channel image array (R, G, B, T)
        :return: preproc_img as a 3 channel image array for model
        """
        img = self.load_image(img_path)

        # process RGB
        rgb_img = img[:, :, :3]

        # --- Channel 1: Retina RGB
        retinex_rgb = cv2.bioinspired_Retina.create((rgb_img.shape[1], rgb_img.shape[0]),
                                                    colorMode=True)
        retinex_rgb.clearBuffers()
        retinex_rgb.run(rgb_img)  # run retina on the input image
        rgb_retinaOut = retinex_rgb.getParvo()  # grab retina outputs
        rgb_retinaOut_gray = cv2.cvtColor(rgb_retinaOut, cv2.COLOR_BGR2GRAY)

        # process Thermal
        tir_img = img[:, :, 3]
        # apply correction to thermal to remove vignetting
        tir_img = self._remove_vignetting(tir_img)

        # --- Channel 2: Retina Thermal
        retinex_tir = cv2.bioinspired_Retina.create([self.SIZE_W, self.SIZE_H])
        retinex_tir.run(tir_img)  # run retina on the input image
        tir_retinaOut = retinex_tir.getParvo()  # grab retina outputs
        # --- Channel 3: Unsharp Mask Thermal
        tir_um = unsharp_mask(tir_img, radius=3, amount=2)
        tir_um = cv2.normalize(tir_um, None, 0, 255, cv2.NORM_MINMAX)
        tir_um = tir_um.astype(np.uint8)

        # stack arrays
        if self.CHANNELS == 3:
            # --- if only_tir flag is True, use only T channel
            if self.only_tir == True:
                preproc_img = np.stack((tir_img, tir_retinaOut, tir_um), axis=2)
            else:
                preproc_img = np.stack((rgb_retinaOut_gray, tir_retinaOut, tir_um), axis=2)

        elif self.CHANNELS == 4:
            # --- Channel 4: Unsharp Mask RGB
            rgb_gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            rgb_um = unsharp_mask(rgb_gray_img, radius=3, amount=2)
            rgb_um = cv2.normalize(rgb_um, None, 0, 255, cv2.NORM_MINMAX)
            rgb_um = rgb_um.astype(np.uint8)

            preproc_img = np.stack((rgb_retinaOut_gray, tir_retinaOut, rgb_um, tir_um), axis=2)
        else:
            raise ValueError(f"Expected either images with 3 or 4 channels for preprocessing with "
                             f" '{self.processing}', got {self.CHANNELS} channels.")
        return preproc_img

    def _stack_array(self, rgb_array, tir_array):
        """
        Stack provided RGB and TIR arrays according to the channel count

        - 4 channels: R + G + B + T
        - 3 channels: grayRGB + T + T

        :param rgb_array: array of shape (:, :, 3)
        :param tir_array: array of shape (:, :)
        :raises ValueError: If channel count isn't 3 or 4
        :return: stacked_array with 3 or 4 channels
        """
        if self.CHANNELS == 3:
            rgb_gray_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            stacked_img = np.stack((rgb_gray_img, tir_array, tir_array), axis=2)
        elif self.CHANNELS == 4:
            stacked_img = np.dstack((rgb_array, tir_array[:, :, np.newaxis]))
        else:
            raise ValueError(f"Expected either images with 3 or 4 channels for preprocessing with "
                             f" '{self.processing}', got {self.CHANNELS} channels.")

        return stacked_img

    @staticmethod
    def _remove_vignetting(image):
        """
        Correction of radial vignetting effect in a provided image

        :param image: 2D array (f.e. thermal image)
        :return: image_corrected: 2D array with corrected radial asymmetry
        """
        # Calculate the distance of each pixel from the center of the image
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        dist = np.sqrt((x - image.shape[1] / 2) ** 2 + (y - image.shape[0] / 2) ** 2)

        # Bin the pixel values based on their radial distance
        num_bins = 100
        bins = np.linspace(0, np.max(dist), num_bins)

        digitized = np.digitize(dist.flat, bins)
        bin_mean = [np.mean(image.flat[digitized == i]) for i in range(1, num_bins)]
        bin_mean = np.array(bin_mean)

        # Fit a function to the mean intensity values for each bin
        fit_func = np.poly1d(np.polyfit(bins[:-1], bin_mean, 10))

        # Divide the original image by the fitted function
        image_corrected = image / fit_func(dist)

        image_corrected = cv2.normalize(image_corrected, None, 0, 255, cv2.NORM_MINMAX)

        return image_corrected.astype(np.uint8)


class MaskProcessor(Base):
    """
    Mask processor
    """
    def __init__(self, config):
        super().__init__(config)
        self.src_mask_dir = Path(self.data_root, config['mask_folder'])

    def load_masks(self, root: str, src_mask_dir: Path = None, overwrite: bool = True):
        """
        Get mask data as a list

        :param root: (str) String denoting dataset - f.e. "train", "test"
        :param src_mask_dir: (Path) source directory for masks before dataset split
                - f.e. "/.../data/masks/"
        :param overwrite: (bool) if True, forces files to be overwritten regardless of existence
        :return: masks: list of 1 channel array masks
        """
        if root not in self.valid_roots:
            raise ValueError(f"Invalid root value. It must be one of '{self.valid_roots}'.")

        # define source directory - user input overrides config definition
        src_mask_dir = Path(src_mask_dir or self.src_mask_dir)
        self.check_folder_exists(src_mask_dir)

        # check if split directory exists and, if so, if it contains masks
        split_mask_dir = Path(self.split_root, root, self.config['mask_folder'])
        if split_mask_dir.is_dir():
            if overwrite is True:
                save = True
                print(f"Saving masks to '{split_mask_dir}'...")
            else:
                save = False
                print(f"Loading masks from previously saved at '{split_mask_dir}'...")
            saved_masks = sorted(list(split_mask_dir.glob("**/*.npy")))
        else:
            save = False
            print(f"Loading masks from '{src_mask_dir}', not saving them...")
            split_mask_dir = src_mask_dir
            saved_masks = []

        # get IDs of masks to be processed
        mask_IDs = read_txt_file(Path(split_mask_dir.parent, root + ".txt"))

        # process masks - if not yet previously done, process; otherwise load from path
        masks = []
        if not saved_masks or len(mask_IDs) != len(saved_masks) or overwrite:
            _logger.info(f"Masks previously not yet or only partially processed. "
                         f"Must load and process from '{src_mask_dir}'...")

            for mask_path in tqdm(sorted([Path(src_mask_dir, m) for m in mask_IDs])):
                mask = self.load_mask(mask_path)
                if save:
                    self.save_npy(npy_arr=mask, dst_dir=split_mask_dir, npy_id=mask_path)
                masks.append(mask)

        else:
            _logger.info(f"Loading masks from '{split_mask_dir}'...")
            # load masks from model train test split path
            for mask_path in tqdm(saved_masks):
                mask = np.load(str(mask_path))
                masks.append(mask)

        return np.array(masks)

    def load_mask(self, mask_path):
        """
        Prepare the mask annotation data to input as y data to model

        :param mask_path: Path to 1 channel mask array
        :return: loaded and resized mask array for model
        """
        # Load mask from source path
        self.check_file_exists(mask_path)
        mask = np.load(str(mask_path))

        # Resize mask to size of choice (of the same aspect ratio)
        height, width = mask.shape
        assert width / height == self.SIZE_W / self.SIZE_H, \
            (f"Aspect ratios of image {(width, height)} and "
             f"dst shape {(self.SIZE_W, self.SIZE_H)} don't match!")

        mask = np.asarray(resize(mask, (self.SIZE_H, self.SIZE_W),
                                 order=0, anti_aliasing=False, preserve_range=True))
        return mask


class ModelLoader(Base):
    """
    Model processor
    """

    def __init__(self, config, model_dir=None):
        """
        Load model and make sure its required inputs match the configuration definitions.

        :param config: configuration dictionary
        :raises AssertionError: If model directory doesn't exist
        :param model_dir: If not none, (Path) path to folder containing model, default uses config
        """
        super().__init__(config)

        # replace configuration definition of model_dir in case method is provided with Path
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(self.config['model_root'], self.config['model_folder'])

        # make sure model directory exists
        assert Path(self.model_dir).exists(), (f"Provided model directory '{self.model_dir}' "
                                               f"does not exist.")
        # load model
        self.model = self.load_model_from_dir()
        self.confirm_model_expected_input()

    def load_model_from_dir(self):
        """
        Load model from provided path with user defined metrics
        :raises IndexError: If no model with suffix .hdf5 in directory
        :return: loaded model
        """
        # find model path in directory
        try:
            model_path = list(Path(self.model_dir).glob("**/*.hdf5"))[0]
        except IndexError:
            _logger.warning(f"Model directory '{self.model_dir}' does not contain a .hdf5 path.")
            sys.exit(-1)

        # define evaluation metrics used in training from config
        METRICS_DICT = {eval(v).__name__: eval(v) for v in self.config['eval']['SM_METRICS'].values()}
        METRICS_DICT[self.config['train']['loss']['name']] = (
            eval(self.config['train']['loss']['function']))

        _logger.info(f"METRICS_DICT:\n{METRICS_DICT.keys()}")

        # load Model (sm Library needs to be imported already)
        _logger.info(f"Loading model from '{model_path}'...")
        with keras.utils.custom_object_scope(METRICS_DICT):
            model = load_model(model_path)

        return model

    def confirm_model_expected_input(self):
        """
        Check what the model expects the shape of the input data to be
        and compare to value in defined in configuration file.

        :raises ValueError: If expected input shape deviates from config-defined input shape
        :raises AssertionError: If input channel count not 3 or 4
        """
        _, size_h, size_w, channels = self.model.layers[0].input_shape[0]

        if size_h != self.SIZE_H or size_w != self.SIZE_W or channels != self.CHANNELS:
            raise ValueError(f"Model expects input images of shape ({size_h}, {size_w},"
                             f" {channels}), but configuration defines shape "
                             f"({self.SIZE_H}, {self.SIZE_W}, {self.CHANNELS}). Mismatch!")
            sys.exit(-1)

        assert channels in [3, 4], (f"Model expects input to have {channels} channels, "
                                    f"but that isn't in the list of viable options [3, 4].")
