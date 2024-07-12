##!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Built-in/Generic
import argparse
from joblib import Parallel, delayed
from pathlib import Path
import sys
import ast

# Other Libraries
# import click
import numpy as np
import cv2
import logging
from skimage.transform import rescale
from skimage.util import img_as_ubyte
from tqdm import tqdm

# Own scripts
from scripts.registration.alignment import calculate_homography, align_image
from scripts.registration.generate_dirtree import generate_dirtree
from scripts.registration.undistortion import undistort_image
from scripts.registration.merging import merge_images

'''
IMAGE REGISTRATION PROCEDURE:
Coordinate the three-step image preprocessing procedure using this overarching script.
Details are provided in the three individual scripts.
'''
_logger = logging.getLogger(__name__)
channels = ["2ch", "4ch"]
sizes = ["640x512", "3750x3000"]


class MatrixManager:
    """
    Helper class for calculating matrices in preparation of preprocessing
    """
    basic_size = None
    scaled_size = None
    rescale_factor = None
    intrinsic_matrix = None
    distortion_coeffs = None
    hom = None
    hom_resc = None

    def __init__(self, calibration_path, homography_dir):
        self.scale_factor_calculator()
        self.calibration_helper(calibration_path)
        self.homography_helper(homography_dir)

    def scale_factor_calculator(self):
        assert len(sizes) == 2, "Scale factor cannot be calculated with less or more than 2 size options!" \
                                "Please redo!"
        self.basic_size = list(map(int, sizes[0].split("x")))
        self.scaled_size = list(map(int, sizes[1].split("x")))

        width_factor = self.scaled_size[0] / self.basic_size[0]
        height_factor = self.scaled_size[1] / self.basic_size[1]
        assert width_factor == height_factor, "Size options do not have the same aspect ratio! Please redo!"

        self.rescale_factor = width_factor

    def calibration_helper(self, calibration_path):
        # prepare step 1: distortion correction
        # --- load calibration file (distortion coefficient and intrinsic matrix)
        calib_data = np.load(str(calibration_path))
        self.intrinsic_matrix = calib_data['intrinsic_matrix']
        self.distortion_coeffs = calib_data['distCoeff']

        _logger.debug(f"Calibration data:"
                      f"\n{self.distortion_coeffs}"
                      f"\n{self.intrinsic_matrix}\n"
                      )

    def homography_helper(self, homography_dir):
        # prepare step 2: calculate homography matrices
        self.hom, self.hom_resc = calculate_homography(scale_factor=self.rescale_factor, sizes=sizes,
                                                       homography_dir=homography_dir)


class ImageManager:
    """
    Helper class to load and manipulate images in ways required for preprocessing
    """
    tir_3ch = None
    tir_resc_3ch = None
    tir_1ch = None
    tir_resc_1ch = None
    rgb = None

    def __init__(self, tir_path, rgb_path, rescale_factor):
        self.tir_path = tir_path
        self.rgb_path = rgb_path
        self.rescale_factor = rescale_factor

        # check to ensure associated RGB image matches TIR
        num = int(tir_path.stem.split("_")[2])
        if num == 999:
            num = 0

        if str(num + 1).zfill(4) in rgb_path.name:
            self.tir_extraction()
            self.rgb_extraction()

    def tir_extraction(self):
        self.tir_3ch = cv2.imread(str(self.tir_path))  # original 3dim array (3 channels)
        tir = self.tir_3ch[:, :, 0]  # convert to 2dim array (0 channels)
        tir_resc = img_as_ubyte(rescale(tir, self.rescale_factor, order=0))
        self.tir_resc_3ch = np.dstack([tir_resc, tir_resc, tir_resc])

        self.tir_1ch = np.expand_dims(tir, axis=2)  # convert to 3dim array (1 channel)
        self.tir_resc_1ch = np.expand_dims(tir_resc, axis=2)

    def rgb_extraction(self):
        self.rgb = cv2.imread(str(self.rgb_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--work-dir', required=True, type=Path,
                        help="Directory for raw RGB and Thermal datasets as well as preprocessing "
                             "results, structured as 'work_dir/raw/images/<set-name>/Thermal OR "
                             "RGB/..._R.JPG OR ....jpg and 'work_dir/merged/<img_channel>/"
                             "<img_size>/images/<set-name>/....npy'."
                             "\nUse script 'generate_dirtree.py' to create.")
    parser.add_argument('-c', '--calibration-path', required=False, type=Path,
                        help='Path to correction calibration file',
                        default=Path("resources", "calibration", "calibration_data.npz"))
    parser.add_argument('-H', '--homography-dir', required=False, type=Path,
                        help="Directory either containing homography matrix (named "
                             "'homography_matrix_<img-size>.npy') or CSV files with RGB and TIR "
                             "featurepoints used to calculate said matrix (named '...RGB...csv' "
                             "and '...TIR...csv').",
                        default=Path("resources", "alignment"))
    parser.add_argument('-ch', '--channels', required=False, default="['2ch', '4ch']",
                        choices=["2ch", "4ch", "['2ch', '4ch']"],
                        help="Choice of different channels of resulting numpy files, with "
                             "4ch = RGBT, 2ch = greyRGB + T")
    parser.add_argument('-sz', '--sizes', required=False, default=['640x512', '3750x3000'],
                        nargs='+',
                        help='List containing size options - original size and to be rescaled size.'
                             'Must be two sizes, in same format and of same aspect ratio.')
    parser.add_argument('-nd', '--no-detailed-outputs', action='store_true', default=False,
                        help="Set flag if outputs should be kept to the minimum. "
                             "This means the aligned RGBs and TIRs aren't overlaid and saved.")
    parser.add_argument('-n', '--njobs', required=True, type=int,
                        help='How many parallel jobs should run?')
    # Combine log level options into one with a default value
    log_levels_group = parser.add_mutually_exclusive_group()
    log_levels_group.add_argument('--quiet', dest='log_level', action='store_const',
                                  const=logging.WARNING, help='Show only warnings.')
    log_levels_group.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                                  const=logging.INFO, help='Show verbose log messages (info).')
    log_levels_group.add_argument('-vv', '--very-verbose', dest='log_level', action='store_const',
                                  const=logging.DEBUG, help='Show detailed log messages (debug).')
    log_levels_group.set_defaults(log_level=logging.INFO)
    return parser.parse_args()


def main(
        work_dir: Path,
        calibration_path: Path,
        homography_dir: Path,
        channels_str: str,
        sizes_list: list,
        no_detailed_outputs: bool,
        njobs: int,
        log_level: int,
):
    global channels
    try:
        channels = ast.literal_eval(channels_str)
    except SyntaxError:
        channels = [channels_str]
    global sizes
    sizes = sizes_list

    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # directories and provided datasets
    dataset_names = [i.stem for i in sorted(list(Path(work_dir, "raw", "images").glob("[!.]*")))]
    assert dataset_names, "The necessary directory tree hasn't yet been created in the provided base folder. " \
                          "Run 'generate_dirtree.py' script and populate the raw/images/.../ folders " \
                          "with RGB and TIR (thermal) images for all datasets!"
    # ensure directory tree (including folders for results) is fully generated for all datasets
    generate_dirtree(dataset_names=dataset_names, base_dir=work_dir, img_channels=channels, img_sizes=sizes)

    helper_data = MatrixManager(calibration_path, homography_dir)

    for dataset_name in dataset_names:
        _logger.info(f"Evaluating {dataset_name}")

        rgb_paths = sorted(list(Path(work_dir, "raw", "images", dataset_name, "RGB").glob("*.jpg")))
        tir_paths = sorted(list(Path(work_dir, "raw", "images", dataset_name, "Thermal").glob("*_R.JPG")))
        _logger.info(f"Loaded {len(rgb_paths)} RGB and {len(tir_paths)} Thermal images...")
        if not tir_paths or not rgb_paths:
            _logger.warning(f"No TIR and/or RGB images in dataset {dataset_name} folder! Ensure images are added to "
                            f"correct folder in the format: TIRs end in '_R.JPG', RGBs in '.jpg'. Skipping...")
            continue

        # check for previous evaluation of current dataset
        prev_eval = [d.stem for d in sorted(list(
            Path(work_dir, "merged", channels[-1], sizes[-1], "images", dataset_name).glob("*.npy")
        ))]
        _logger.debug(f"Previously evaluated: {len(prev_eval)} out of {len(tir_paths)}")

        # loop through images in parallel
        # - if statement allows images to be skipped that have already been evaluated
        Parallel(n_jobs=njobs)(
            delayed(register_image_pairs)(tir_path, rgb_paths[idx], work_dir, dataset_name,
                                          helper_data, no_detailed_outputs)
            for idx, tir_path in enumerate(tqdm(tir_paths))
            if tir_path.stem not in prev_eval
        )

        _logger.info(f"Finished evaluating {dataset_name}")


def register_image_pairs(tir_path: Path, rgb_path: Path, work_dir: Path, dataset_name: str,
                         helper_data, no_details: bool):
    """
    Process individual image pairs by 1. undistorting RGBs, 2. aligning RGBs with TIRs and 3. merging the results

    :param tir_path: (Path) directory of TIR image
    :param rgb_path: (Path) directory of RGB image
    :param helper_data: (class object) matrices and other data required for preprocessing
    :param work_dir: (Path) working directory
    :param dataset_name: (Path) dataset name
    :param no_details: (bool) if True, don't save extra images in alignment
    """

    img_data = ImageManager(tir_path, rgb_path, helper_data.rescale_factor)

    if img_data.rgb is not None:
        # 1. Undistort RGB
        undistorted_rgb = undistort_image(img_stem=rgb_path.stem, img=img_data.rgb,
                                          intrin_mat=helper_data.intrinsic_matrix,
                                          dist_coeff=helper_data.distortion_coeffs,
                                          out_dir=Path(work_dir, "raw", "images", dataset_name, "RGB_undistorted"))

        # 2. Align undistorted RGB with original sized and rescaled TIR
        # original size (i.e. 640x512)
        aligned_rgb = align_image(tir_stem=tir_path.stem, rgb_stem=rgb_path.stem, M_hom=helper_data.hom,
                                  tir=img_data.tir_3ch, rgb=undistorted_rgb, no_details=no_details,
                                  out_dir=Path(work_dir, "raw", "images", dataset_name, "RGB_aligned", sizes[0]))
        # rescaled size (i.e. 3750x3000)
        aligned_rgb_resc = align_image(tir_stem=tir_path.stem, rgb_stem=rgb_path.stem, M_hom=helper_data.hom_resc,
                                       tir=img_data.tir_resc_3ch, rgb=undistorted_rgb, no_details=no_details,
                                       out_dir=Path(work_dir, "raw", "images", dataset_name, "RGB_aligned", sizes[1]))

        # 3. Merge aligned RGB and TIR images to generate numpy files
        # original size (i.e. 640x512)
        merge_images(tir=img_data.tir_1ch, rgb=aligned_rgb, work_dir=work_dir, channels=channels,
                     end_of_path=Path(sizes[0], "images", dataset_name, tir_path.stem + ".npy"))
        # rescaled size (i.e. 3750x3000)
        merge_images(tir=img_data.tir_resc_1ch, rgb=aligned_rgb_resc, work_dir=work_dir, channels=channels,
                     end_of_path=Path(sizes[1], "images", dataset_name, tir_path.stem + ".npy"))

    else:
        _logger.warning(f"No matching RGB found for {tir_path}\n"
                        f"---skipping {tir_path.name} TIR and {rgb_path.name} RGB.")


if __name__ == "__main__":
    args = parse_args()
    main(
        Path(args.work_dir),
        Path(args.calibration_path),
        Path(args.homography_dir),
        args.channels,
        args.sizes,
        args.no_detailed_outputs,
        args.njobs,
        args.log_level,
    )
