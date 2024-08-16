#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to create and save multiclass segmentation masks
from json annotation files
"""
# import built-in dependencies
import argparse
import difflib
import json
import logging
import sys
from math import cos, sin, radians
from pathlib import Path

# import external dependencies
import cv2
from matplotlib.colors import ListedColormap
from PIL import Image
import numpy as np
import pandas
import pandas as pd
from skimage.draw import polygon2mask
from tqdm import tqdm

from tufseg.scripts.configuration import init_temp_conf, update_conf
config = init_temp_conf(delete_existing=True)
# --------------------------------------
_logger = logging.getLogger("generate_segm_masks")

labels = config['data']['masks']['labels']
IMG_SHAPE = (None, None)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generating segmentation masks from json annotation files'
    )
    parser.add_argument('-j', '--json-dir', dest='json_dir',
                        required=True, type=str,
                        help='Directory containing JSON annotation files to '
                             'convert into segmentation masks')
    parser.add_argument('-i', '--image-dir', dest='img_dir',
                        required=True, type=str,
                        help='Path to "images" directory containing image '
                             'dataset subdirs with names matching the '
                             'annotation files.')
    parser.add_argument('-sv', '--save-for-view', dest='save_for_view',
                        action='store_true',
                        help='Whether to additionally save segmentation masks '
                             'for viewing in .png format.')

    # Combine log level options into one with a default value
    log_levels_group = parser.add_mutually_exclusive_group()
    log_levels_group.add_argument('--quiet', dest='log_level',
                                  action='store_const', const=logging.WARNING,
                                  help='Show only warnings.')
    log_levels_group.add_argument('-v', '--verbose', dest='log_level',
                                  action='store_const', const=logging.INFO,
                                  help='Show verbose log messages (info).')
    log_levels_group.add_argument('-vv', '--very-verbose', dest='log_level',
                                  action='store_const', const=logging.DEBUG,
                                  help='Show detailed log messages (debug).')
    log_levels_group.set_defaults(log_level=logging.WARNING)
    return parser.parse_args()


def main(
    json_dir: Path,
    img_dir: Path,
    save_for_view: bool = False,
    log_level=logging.WARNING
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt='%Y-%m-%d %H:%M',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main_args = locals()
    for k, v in main_args.items():
        _logger.info(f'Parsed user argument - {k}:\t{v}')

    create_segm_mask_from_json(
        json_dir=Path(json_dir),
        img_dir=Path(img_dir),
        save_for_view=save_for_view
    )

    # update config file
    main_args = {
        "data_root": str(Path(img_dir).parent),
        "img_folder": Path(img_dir).name,
        "anno_root": json_dir
    }
    _logger.info(f"Updating configuration file with:\n{main_args}")
    update_conf(conf=config, params=main_args)


def create_segm_mask_from_json(json_dir: Path, img_dir: Path,
                               save_for_view: bool):
    """
    Coordinate segmentation mask creation from annotation json files

    :param json_dir: (Path) directory containing .json annotation files
    :param img_dir: (Path) directory containing dataset subdirs
                    (such as KA_01, MU_01, ...)
    :param save_for_view: (bool) whether to save masks for viewing
                            in png format to mask dir
    :return: Segmentation masks saved to same folder as image directory
    """
    # create destination directory if it doesn't yet exist
    masks_dir = Path(img_dir.parent, config['mask_folder'])
    masks_dir.mkdir(parents=True, exist_ok=True)

    labels_df_list = []

    # Loop through all datasets
    break_bool = False
    for dataset_path in img_dir.glob('*'):
        # Check if provided directory contains numpys instead of subdirs
        # i.e. images/DJI_....npy
        if dataset_path.is_file():
            _logger.info(
                f"Directory '{img_dir}' contains .npy files instead of "
                f"further image subdirectories. Using '{img_dir.name}' folder "
                f"as dataset_path!"
            )
            dataset_path = img_dir
            if len(list(json_dir.glob("*.json"))) != 1:
                raise ValueError(
                    f"Annotation file allocation not possible with one dataset"
                    f" and multiple annotation files in {json_dir}!"
                )

            json_path = json_dir.glob("*.json")[0]
            break_bool = True

        else:
            # Find annotation file to match dataset, skip if it doesn't exist
            json_path = Path(json_dir, f'{dataset_path.stem}_json.json')
            if not json_path.exists():
                # check for close matching files
                matches = difflib.get_close_matches(
                    json_path.name,
                    [f.name for f in json_dir.glob("*.json")],
                    cutoff=0.95
                )
                if matches:
                    _logger.info(
                        f"Found annotations '{matches[0]}' to "
                        f"replace '{json_path.name}'"
                    )
                    json_path = Path(json_dir, matches[0])
                else:
                    _logger.warning(
                        f"ANNOTATION FILE '{json_path.name}' DOES NOT EXIST, "
                        f"skipping!"
                    )
                    continue

        dataset_labels_df = extract_anno_df(
            json_path=json_path,
            set_dir=dataset_path
        )
        labels_df_list.append(dataset_labels_df)

        if break_bool:
            break

    labels_df = pd.concat(labels_df_list, ignore_index=True)

    anno_count = labels_df['label'].value_counts()
    _logger.info(f"Complete annotation overview:\n{anno_count}")

    create_masks(
        anno_df=labels_df,
        dst_dir=masks_dir,
        save_for_viewing=save_for_view
    )


def extract_anno_df(json_path: Path, set_dir: Path,
                    descriptor="thermal_objects"):
    """
    This function loads the JSON file created with the VGG annotator
    and returns the annotations within.

    The returned DataFrame has the structure:
      #,  img_id,                        label,      label_id, region_coords
      0  images/MU_01/DJI_0_0001_R.npy  car (warm)  2         [(217, 191), ...]
      1  images/MU_01/DJI_0_0001_R.npy  person      6         [(11, 13), ...]

    with region_coords being a list of (y, x) coordinate pairs of the
    annotation region perimeter

    :param json_path: (str / Path object) Location of JSON annotation file
    :param set_dir: (str / Path object) Directory containing corresponding
                    numpy image files
    :param descriptor: (str) Name of the dict key in json's region_attributes
                for the annotation class (defined by user during annotation)
    :return: Pandas DataFrame containing annotation information in the form
         ['img_id': str, 'label': str, 'label_id': int, 'region_coords': list]
    """
    global IMG_SHAPE

    with open(json_path) as f:
        anno_img_data = json.load(f)
    _logger.info(f"JSON file '{json_path.name}' successfully loaded")

    skipped = 0
    df = pandas.DataFrame(
        columns=['img_id', 'label', 'label_id', 'region_coords']
    )

    for idx, anno_img in enumerate(anno_img_data.values()):

        img_stem = Path(anno_img['filename']).stem
        npy_path = Path(set_dir, f'{img_stem}.npy')

        # Check the image actually exists
        if not npy_path.exists():
            _logger.warning(
                f"{set_dir.name} - Numpy image file missing, "
                f"skipping: '{npy_path}'"
            )
            skipped += 1
            continue
        # Get width and height of image
        if IMG_SHAPE == (None, None):
            file = np.load(npy_path)
            IMG_SHAPE = file.shape[:2]

        # Loop through all annotations in image
        regions = anno_img["regions"]
        for region in regions:
            anno = region['shape_attributes']

            # handle each shape differently
            if (anno['name'] == 'polygon') or (anno['name'] == 'polyline'):
                px = anno["all_points_x"]
                py = anno["all_points_y"]
            elif anno['name'] == 'rect':
                # Coords are bottom left corner of rectangle,
                # as CS origin is in top left corner for images
                p_x = anno['x']
                p_y = anno['y']
                rect_w = anno['width']
                rect_h = anno['height']
                px = [p_x, p_x + rect_w, p_x + rect_w, p_x]
                py = [p_y, p_y, p_y - rect_h, p_y - rect_h]
            elif anno['name'] == 'circle':
                center_x = anno['cx']
                center_y = anno['cy']
                r = round(anno['r'])
                # calculate circle border x and y points to create polygon
                px = []
                py = []
                for phi in range(0, 380, 60):
                    px.append(round(r * cos(radians(phi)) + center_x))
                    py.append(round(r * sin(radians(phi)) + center_y))
            else:
                _logger.warning(
                    f"{set_dir.name} - {img_stem} - "
                    f"unsupported shape {anno['name']}!"
                )
                raise KeyError(
                    f'Unsupported shape {anno["name"]} found in {img_stem}'
                )

            # define region coordinates as a list of (y, x) pair coordinates
            region_coords = list(zip(py, px))

            # Find region category id, ensure it matches a valid category
            region_label = region['region_attributes'][descriptor]
            try:
                region_label_id = labels.index(region_label)
            except ValueError:
                try:
                    region_label = difflib.get_close_matches(region_label,
                                                             labels)[0]
                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"An annotation in image {anno_img['filename']} of "
                        f"'{json_path.name}'\nhas the label '{region_label}',"
                        f"which does not correspond to any of the allowed "
                        f"categories:\n{labels}\nPlease correct!"
                    ) from e

                region_label_id = labels.index(region_label)

            # add annotation information as a new row to the pandas dataframe
            df.loc[len(df)] = [
                str(Path(*npy_path.parts[-2:])),  # image, i.e. MU_02/DJI...npy
                region_label,   # label / class of the current region
                region_label_id,  # ID of label / class of the current region
                region_coords  # coordinates of current region
            ]

    _logger.info(
        f"{set_dir.name} - Skipped {skipped}/{len(anno_img_data)} files in "
        f"directory '{set_dir}'"
    )

    anno_count_in_set = df['label'].value_counts()
    _logger.debug(
        f"{set_dir.name} - Annotation overview:\n{anno_count_in_set}"
    )

    return df


def create_masks(anno_df, dst_dir, save_for_viewing=False):
    """
    Create multiclass segmentation masks for each image from
    information provided by annotation dataframe and save to
    destination directory

    :param anno_df: (pd.DataFrame) Dataframe with img_id, label, label_id,
                    region_coords information on all annotations
    :param dst_dir: (Path) destination directory for masks
    :param save_for_viewing: (bool) If True, saves segmentation masks as
                            coloured pngs for viewing
    """
    global IMG_SHAPE
    ANNO_IMG_SHAPE = config['data']['ANNO_IMG_SHAPE']
    rescale_factor = IMG_SHAPE[0] / ANNO_IMG_SHAPE[0]

    _logger.info(f"Rescale factor is {rescale_factor}")

    # get a list of all images
    unique_image_ids = anno_df['img_id'].unique()

    _logger.info(f"Creating masks and saving to '{dst_dir}'...")
    for unique_img_id in tqdm(unique_image_ids):
        _logger.debug(f"Creating mask for '{unique_img_id}'...")

        # extract rows from existing dataframe for current image
        img_df = anno_df[anno_df['img_id'] == unique_img_id]

        # get all the image's annotation polygon coordinates and labels IDs
        coord_pair_list = img_df['region_coords'].tolist()
        label_id_list = img_df['label_id'].tolist()

        # create mask for each anno, assign color, combine to multichannel mask
        combined_mc_mask = np.zeros(ANNO_IMG_SHAPE, dtype=int)

        for idx, coord_pair in enumerate(coord_pair_list):
            polygon = np.array(coord_pair)
            mask = polygon2mask(ANNO_IMG_SHAPE, polygon)    # bool mask

            # create mask with background = 0 and annotations = label_id + 1
            mc_mask = np.where(
                mask == True,   # noqa: E712
                int(label_id_list[idx]) + 1,
                0
            ).astype(np.uint8)

            # if a pixel was annotated >1 times, choose the first class
            combined_mc_mask = np.where(
                np.logical_and(combined_mc_mask != 0, mc_mask != 0),
                combined_mc_mask,
                combined_mc_mask + mc_mask
            )

        combined_mc_mask = combined_mc_mask.astype(np.uint8)
        _logger.debug(
            f"Mask for '{unique_img_id}' has labels: "
            f"{np.unique(combined_mc_mask)}"
        )

        # save mask as png image
        if rescale_factor != 1:
            combined_mc_mask = cv2.resize(
                combined_mc_mask,
                dsize=tuple(reversed(IMG_SHAPE)),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)

        # define path, create parent directory (dataset like MU_01)
        mask_set_dir = Path(dst_dir, Path(unique_img_id)).parent
        mask_set_dir.mkdir(parents=True, exist_ok=True)

        mask_path = Path(dst_dir, Path(unique_img_id))
        _logger.debug(f"Saving mask to {mask_path}")

        # save for later use in training etc
        np.save(mask_path, combined_mc_mask)

        if save_for_viewing:
            mask_png_path = mask_path.with_suffix(".png")
            # save with coloured annotations
            colormap = ListedColormap(config['data']['masks']['custom_colors'])
            col_mask_img = Image.fromarray(
                (colormap(combined_mc_mask) * 255).astype(np.uint8)
            )
            col_mask_img.save(str(mask_png_path))


if __name__ == "__main__":
    args = parse_args()
    main(
        img_dir=args.img_dir,
        json_dir=args.json_dir,
        save_for_view=args.save_for_view,
        log_level=args.log_level
    )
