#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to randomly split test and train datasets by user defined value
and generate a directory tree including all relevant data for training
and testing
"""
# import built-in dependencies
import argparse
import difflib
from pathlib import Path
import json
import logging
import re
import sys
from typing import Union

# import external dependencies
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tufseg.configuration import init_temp_conf, update_conf
config = init_temp_conf()
# --------------------------------------
_logger = logging.getLogger("train_test_split")

descriptor = config['data']['annotations']['descriptor']
labels = config['data']['annotations']['classes']

SRC_DIR = Path
DST_DIR: Path = None

CATEGORY_IDs = []
DATASET_NAMES = []


def parse_args():
    parser = argparse.ArgumentParser(description='Split data into test and train sets.')
    parser.add_argument('-src', '--src-dir', dest='source_dir',
                        help='Path to directory containing images and segmentation masks '
                             'subfolders. If none is provided, the directory from the config '
                             'will be used.')
    parser.add_argument('-dst', '--dst-dir', dest='destination_dir',
                        help='Path to destination directory for saving train / test split '
                             'If none is provided, the train.txt and test.txt files will '
                             'be saved to the source directory.')
    parser.add_argument('--test-size', type=float, default=0.2, dest='test_size',
                        help="Approximate size the test dataset should have (decimal)")
    parser.add_argument('--random-state', type=int, default=1, dest='random_state',
                        help="Number to generate reproducible random statistical split")
    parser.add_argument('--no-anno-check', action='store_true', dest='no_anno_check',
                        help='Include to NOT take annotation distribution into account in split')
    parser.add_argument('--no-class-check', action='store_true', dest='no_class_check',
                        help='Include to NOT take class distribution into account in split')
    parser.add_argument('--no-set-check', action='store_true', dest='no_set_check',
                        help='Include to NOT take dataset distribution into account in split')

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
        source_dir: Union[None, Path],
        destination_dir: Union[None, Path],
        test_size: float = 0.2,
        random_state: int = 1,
        check_anno: bool = True,
        check_class: bool = True,
        check_set: bool = True,
        log_level=logging.WARNING
):
    """
    Split all annotated data into train and test datasets according to user-provided inputs.
    """
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_args = locals()
    for k, v in main_args.items():
        _logger.info(f'Parsed user argument - {k}:\t{v}')


    global SRC_DIR
    SRC_DIR = Path(source_dir or config['data_root'])
    assert SRC_DIR.is_dir(), f"Provided source directory {SRC_DIR} is not an existing folder!"

    masks_dir = Path(SRC_DIR, config['mask_folder'])
    jsons_dir = Path(config['anno_root'])

    # Update global variable for the category IDs for later cross-referencing
    global CATEGORY_IDs
    CATEGORY_IDs = [idx for idx, i in enumerate(labels)]
    _logger.debug(f"{len(CATEGORY_IDs)} different categories annotated: {labels}")
    # Update global variable for dataset folder names
    global DATASET_NAMES
    DATASET_NAMES = [dn.name for dn in sorted(masks_dir.glob("*"))
                     if re.match(r'^[a-zA-Z]+_\d+$', dn.name)]
    _logger.debug(f"{len(DATASET_NAMES)} different datasets provided: {DATASET_NAMES}")

    # list of annotated image names including dataset folder paths for unambiguous identification
    # --- Don't use images_dir here because maybe not all the images were annotated!
    image_identifiers = [Path(i.parent.name, i.name)
                         for i in sorted(masks_dir.glob("**/*.npy"))]

    # Get list of annotation categories in each image for split
    annotations_per_img = get_all_annotations(json_dir=jsons_dir)

    # Split all data into train and test datasets
    images_train, images_test = split_train_test(
        images=image_identifiers, annotations=annotations_per_img,
        test_size=test_size, random_state=random_state,
        check_anno_distrib=check_anno, check_class_distrib=check_class,
        check_set_distrib=check_set
    )

    # Create the split directory tree
    if destination_dir:
        # update config file
        update_conf(conf=config, params={"split_root": destination_dir})

        global DST_DIR
        DST_DIR = Path(destination_dir)
        if not DST_DIR.exists():
            DST_DIR.mkdir(parents=True)

        # Build directory tree to populate during train preprocessing
        build_dirtree()

    # Save split by saving image identifiers into separate text files
    save_train_test_split(images_train, images_test)


def get_all_annotations(json_dir):
    """
    Get a list of annotation category IDs for all images.

    :param json_dir: (Path) directory to all VIA JSON files
    :return: annos_per_img - list of lists with a sublist for each image
    """
    _logger.info("Extracting annotation information from JSONs...")
    annos_per_img = []
    for json_path in tqdm(Path(json_dir).glob("*.json")):
        with open(json_path) as f:
            anno_data = json.load(f)

        for idx, anno_img in enumerate(anno_data.values()):
            regions = anno_img["regions"]

            img_annos = []
            for region in regions:

                # Find region category id while ensuring it matches the list of possible categories
                region_label = region['region_attributes'][descriptor]

                try:
                    region_label_id = labels.index(region_label)
                except ValueError:
                    region_label = difflib.get_close_matches(region_label, labels)[0]
                    region_label_id = labels.index(region_label)

                # Add category ID to image category list
                img_annos.append(region_label_id)

            # Add image annotation list of category IDs to overall list
            annos_per_img.append(img_annos)

    return annos_per_img


def split_train_test(
        images: list, annotations: list, test_size: float, random_state=1,
        check_anno_distrib=True, check_class_distrib=True, check_set_distrib=True
):
    """
    Split list of images and matching annotations into test / train datasets
    according to the size provided by the user and checks according to user choices.

    Args:
        images: (list) List containing image identifying paths
        annotations: (list) List of sublists annotation category IDs - one for each image
        test_size: (float) percentage of test dataset i.e. 0.2 would make a 20%-80% split
        random_state: (int) randomness value - set to value to be able to reproduce split
        check_anno_distrib: (bool) set to True to include an annotation distribution check
        check_class_distrib: (bool) set to True to include a class label distribution check
        check_set_distrib: (bool) set to True to include a dataset distribution check

    Returns:
        images_train - list of train images,
        images_test - list of test images
    """
    # Use sklearn to split images into two lists
    images_train, images_test, annotations_train, annotations_test = \
        train_test_split(images, annotations, test_size=test_size, random_state=random_state)

    # Calculate the distribution of annotations in the train and test sets
    train_counts = sum(len(sublist) for sublist in annotations_train)
    test_counts = sum(len(sublist) for sublist in annotations_test)

    _logger.debug(overview(images, test_size, train_counts, test_counts, images_train, images_test))

    if check_anno_distrib:
        # Check if the distribution of annotations is too skewed - allow for +/- 5%
        true_test_size = test_counts / sum(len(sublist) for sublist in annotations)
        if true_test_size < np.maximum(0, test_size - 0.05) or \
                true_test_size > np.minimum(1, test_size + 0.05):
            _logger.warning(f"Current split is too skewed based on annotation counts! "
                            f"Reshuffling (random state: {random_state + 1})...")
            return split_train_test(images, annotations, test_size, random_state=random_state + 1)

    if check_class_distrib:
        # Check if all classes are represented in test and train datasets
        missing_ids_train = [cat_id for cat_id in CATEGORY_IDs if
                             not any(cat_id in sublist for sublist in annotations_train)]
        missing_ids_test = [cat_id for cat_id in CATEGORY_IDs if
                            not any(cat_id in sublist for sublist in annotations_test)]
        if missing_ids_test or missing_ids_train:
            _logger.warning(f"Current split doesn't have all class categories in both "
                            f"train/test! Reshuffling (random state: {random_state + 1})...")
            return split_train_test(images, annotations, test_size, random_state=random_state + 1)

    if check_set_distrib:
        # Check if all datasets are represented in test and train datasets
        missing_dataset_train = [dataset for dataset in DATASET_NAMES if dataset not in
                                 [img_path.parent.name for img_path in images_train]]
        missing_dataset_test = [dataset for dataset in DATASET_NAMES if dataset not in
                                [img_path.parent.name for img_path in images_test]]
        if missing_dataset_test or missing_dataset_train:
            _logger.warning(f"Current split doesn't have images from all datasets in both "
                            f"train/test. Reshuffling (random state: {random_state + 1})...")
            return split_train_test(images, annotations, test_size, random_state=random_state + 3)

    _logger.info(f"Suitable split found at random_state {random_state}.")
    _logger.info(overview(images, test_size, train_counts, test_counts, images_train, images_test))
    return images_train, images_test


def overview(images, test_size, train_counts, test_counts, images_train, images_test):
    print_message = \
        f"Dataset split with user input {test_size * 100}% test, " \
        f"{len(images)} images and {train_counts + test_counts} annotations:\n" \
        f"- train:\t{len(images_train)} images " \
        f"({round(len(images_train) / len(images) * 100, 1)}%), " \
        f"{train_counts} annotations " \
        f"({round(train_counts / (train_counts + test_counts) * 100, 1)}%)" \
        f"\n- test:\t\t{len(images_test)} images " \
        f"({round(len(images_test) / len(images) * 100, 1)}%)," \
        f" {test_counts} annotations " \
        f"({round(test_counts / (train_counts + test_counts) * 100, 1)}%)"

    return print_message


def build_dirtree():
    """
    Create directory tree for test train split using dataset folder names
    """
    directory_list = [
        Path(DST_DIR, s, i, d) for s in ["train", "test"] for i in ["images", "masks"]
        for d in DATASET_NAMES
    ]

    # create directories
    for directory in directory_list:
        directory.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Built the test/train directory tree at '{DST_DIR}'")


def save_train_test_split(images_train: list, images_test: list):
    """
    Save lists of image paths ("dataset_folder/img_name.npy")
    of train and test datasets in text files to extract in training.

    Args:
        images_train: (list) image identifiers for training dataset
        images_test: (list) image identifiers for testing dataset

    Returns:
        Saved "train.txt" and "test.txt"
    """
    # convert Path objects to strings for saving
    images_train = [str(i) for i in images_train]
    images_test = [str(i) for i in images_test]

    # define textfile locations to save to
    if DST_DIR:
        train_txt = Path(DST_DIR, "train", "train.txt")
        test_txt = Path(DST_DIR, "test", "test.txt")
    else:
        train_txt = Path(SRC_DIR, "train.txt")
        test_txt = Path(SRC_DIR, "test.txt")

    # save to text files
    with open(train_txt, "w") as f:
        f.write('\n'.join(images_train))

    with open(test_txt, "w") as f:
        f.write('\n'.join(images_test))

    _logger.info(f"Train and test image path identifiers saved to "
                 f"'{train_txt}' and '{test_txt}'.")


if __name__ == "__main__":
    args = parse_args()
    main(
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        check_anno=not args.no_anno_check,
        check_class=not args.no_class_check,
        check_set=not args.no_set_check,
        log_level=args.log_level
    )
