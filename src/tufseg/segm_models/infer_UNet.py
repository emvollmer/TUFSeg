#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer using trained UNet model
"""
# import built-in dependencies
import argparse
import logging
import os
import sys
from pathlib import Path
import warnings

# import external dependencies
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from tufseg.segm_models._utils import (
    configure_logging, ModelLoader, ImageProcessor, MaskProcessor
)
from tufseg.configuration import read_conf

# set matplotlib logger to a higher level to suppress debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.ERROR)
# --------------------------------------
_logger = logging.getLogger('infer')

config: dict


def parse_args():
    """Get user required arguments"""
    parser = argparse.ArgumentParser(description='Inferring on image with model')
    parser.add_argument('-model', '--model-dir', dest='model_dir', required=True,
                        help='Path to model directory containing model to be used for inference.')
    parser.add_argument('-img', '--img-pth', dest='img_path', required=True,
                        help='Path to numpy image to be inferred upon.')
    parser.add_argument('-mask', '--mask-pth', dest='mask_path',
                        help='OPTIONAL - Path to true segmentation mask for comparison with inference.'
                             '(If mask exists in folder structure, it will be loaded automatically).')
    parser.add_argument('-display', dest='display', action='store_true',
                        help='Whether to plot the resulting prediction to the console.')
    parser.add_argument('-save', dest='save', action='store_true',
                        help='Whether to save the resulting prediction to the model folder.')

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
        model_dir: Path,
        img_path: Path,
        mask_path: Path = None,
        display: bool = False,
        save: bool = False,
        log_level=logging.WARNING
):
    """
    Coordinate inference
    """
    global config
    config = read_conf(Path(model_dir, "run_config.json"))

    configure_logging(_logger, log_level)

    # log user provided inputs
    main_args = locals()
    for k, v in main_args.items():
        _logger.info(f'Parsed user argument - {k}:\t{v}')

    # load model
    model_loader = ModelLoader(config, model_dir)
    model = model_loader.model

    # preprocess image for inference depending on what the model expects / is defined in the config
    img_path = Path(img_path)
    img_processor = ImageProcessor(config)
    proc_img = img_processor.proc_method(img_path)

    # predict on image with model
    _logger.info(f"Predicting on image '{img_path.name}' with model '{config['model']['type']}'")
    infer_mask = predict(
        model=model, proc_img=proc_img
    )

    # check if ground truth (segmented mask) exists in folder structure
    try:
        img_folder_index = img_path.parts.index(config['img_folder'])

        data_root = Path(*img_path.parts[:img_folder_index])
        mask_name = Path(*img_path.parts[img_folder_index + 1:])

        match_mask_path = Path(data_root, config['mask_folder'], mask_name)
        if match_mask_path.is_file():
            _logger.info(f"Ground truth segmentation mask found for image '{img_path.name}' at '{match_mask_path}'")
            mask_path = match_mask_path
    except ValueError:
        pass

    # load ground truth (segmented mask) if provided
    mask = None
    if mask_path is not None:
        mask_path = Path(mask_path)
        assert mask_path.name == img_path.name, \
            f"Provided mask '{mask_path.name}' does not match image '{img_path.name}'!"

        mask_processor = MaskProcessor(config)
        mask = mask_processor.load_mask(mask_path)
        _logger.info(f"Using ground truth segmentation mask at '{mask_path}' for comparison.")

    # save or display results if desired by user
    if display or save:
        save_dir = None
        if save:
            save_dir = Path(Path(model_dir), "predictions")
            save_dir.mkdir(parents=True, exist_ok=True)

        plot(
            src_img_path=Path(img_path), proc_img=proc_img,
            infer_mask=infer_mask, true_mask=mask,
            save_dir=save_dir, display=display
        )


def predict(model, proc_img):
    """
    Infer on image provided by user

    :param model: previously trained and loaded model
    :param proc_img: numpy array of processed image to be inferred upon with model
    :return class_predictions_rgb - 3 channel mask of predictions
    """
    # predict on preprocessed image -> shape: (1, SIZE_H, SIZE_W, NUM_CLASSES)
    prediction = model.predict(
        np.expand_dims(proc_img, axis=0),
        verbose=1
    )
    class_predictions = np.argmax(prediction, axis=-1)  # max across NUM_CLASSES channels -> (1, SIZE_H, SIZE_W)
    class_predictions = np.squeeze(class_predictions)   # remove unnecessary dim -> (SIZE_H, SIZE_W)

    return class_predictions


def plot(src_img_path: Path, proc_img: np.ndarray,
         infer_mask: np.ndarray, true_mask: np.ndarray = None,
         save_dir: Path = None, display: bool = False):
    """
    Display or save predictions plotted side-by-side next
    to the original RGB and Thermal images.

    :param src_img_path: Path to original numpy array of basic 4 channel image
    :param proc_img: numpy array of image processed according to config 'processed' value
    :param infer_mask: numpy array of predictions inferred by model
    :param true_mask: If array, display ground truth mask next to predicted mask
    :param save_dir: If Path, save plotted side-by-side and predictions themselves
    :param display: If True, display plotted side-by-side to console
    """
    # Load original image
    img = np.load(str(src_img_path))

    # create coloured mask of predictions
    colormap = ListedColormap(config['data']['masks']['custom_colors'])
    rgb_infer_mask = (colormap(infer_mask) * 255).astype(np.uint8)[:, :, 0:3]

    # Create 3x1 subfigures
    num_channels = proc_img.shape[-1]
    row_headings = ["Original four channel image", "Processed image channels used as model input", "Segmentation mask"]
    
    fig = plt.figure(constrained_layout=True, figsize=(num_channels*6, 18))
    fig.suptitle(f'Inference Results - {Path(src_img_path.parent.name, src_img_path.name)}',
                 fontsize=22)

    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(row_headings[row], fontsize=18, weight='bold')

        if row == 0:
            axs = subfig.subplots(nrows=1, ncols=2)
            # first row: original image data
            axs[0].imshow(img[:, :, :3])
            axs[0].set_title("RGB", fontsize=15)
                    
            axs[1].imshow(img[:, :, 3], cmap='gray')
            axs[1].set_title("Thermal", fontsize=15)
                        
        elif row == 1:
            axs = subfig.subplots(nrows=1, ncols=num_channels)
            # second row: preprocessed image channels
            for i in range(num_channels):
                axs[i].imshow(proc_img[:, :, i], cmap='gray')
                axs[i].set_title(f"Channel {i + 1}", fontsize=15)
                    
        elif row == 2:
            # third row: class predictions output with color legend
            labels = config['data']['masks']['labels']
            legend_patches = [mpatches.Patch(color=colormap(i+1), label=labels[i])
                              for i in range(len(labels))]

            if true_mask is not None:
                axs = subfig.subplots(nrows=1, ncols=2)
                axs[0].imshow((colormap(true_mask) * 255).astype(np.uint8)[:, :, 0:3])
                axs[0].set_title("True mask based on annotations", fontsize=15)
                axs[1].imshow(rgb_infer_mask)
                axs[1].set_title("Predicted mask", fontsize=15)

                axs[1].legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1),
                              fontsize=14)
            else:
                axs = subfig.subplots(nrows=1, ncols=1)
                axs.imshow(rgb_infer_mask)

                axs.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1),
                           fontsize=14)

    if display:
        _logger.info("Displaying prediction outputs to console...")
        plt.show()

    if save_dir:
        save_dir = Path(save_dir)
        save_folder_dir = Path(save_dir, src_img_path.parent.name)
        if not save_folder_dir.exists():
            save_folder_dir.mkdir(parents=True)

        # save prediction as numpy and png
        dst_pth = str(Path(save_folder_dir, src_img_path.name))
        np.save(dst_pth, infer_mask)

        dst_pred_pth = Path(save_folder_dir, src_img_path.stem + ".png")
        rgb_infer_img = Image.fromarray(rgb_infer_mask)
        rgb_infer_img.save(dst_pred_pth)

        # save matplotlib side-by-side comparison
        dst_overview_pth = Path(save_folder_dir, src_img_path.stem + "_overview.png")
        fig.savefig(dst_overview_pth)
        
        _logger.info(f"Saved prediction outputs to '{dst_pth}'.")


if __name__ == "__main__":
    args = parse_args()
    main(
        model_dir=args.model_dir,
        img_path=args.img_path,
        mask_path=args.mask_path,
        display=args.display,
        save=args.save,
        log_level=args.log_level
    )
