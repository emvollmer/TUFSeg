##!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import csv
from pathlib import Path
import numpy as np
import cv2

'''
Helper functions for STEP 2 in the image preprocessing procedure:
Register RGB images onto thermal images via homography.
Manually selected matching feature point pairs are used, which are
stored in csv files.
RGB images are 4000*3000, thermal images are 640*512.
NOTE: Because of the size difference we SCALE the thermal (destination)
points up to match that of the RGB.

Combining the images comes in step 3.
'''
_logger = logging.getLogger(__name__)


def calculate_homography(scale_factor: float, sizes: list,
                         homography_dir: Path):
    """
    Calculate homography matrices required for aligning images.

    Args:
        scale_factor: Factor to scale up to required format
                      (here: 640x512 to 3750x3000)
        sizes: List of sizes that are chosen from for homography calculation
        homography_dir: Directory containing either homography matrices
                        or CSV files for RGB and TIR featurepoints
                        to calculate the homography

    Returns:
        hom, hom_resc: homography matrix, rescaled homography matrix
    """
    # check if homography was previously saved, otherwise calculate it
    hom_path = Path(homography_dir, f"homography_matrix_{sizes[0]}.npz")
    hom_resc_path = Path(homography_dir, f"homography_matrix_{sizes[1]}.npz")

    if Path.exists(hom_resc_path):
        hom = np.load(str(hom_path))["M_homography"]
        hom_resc = np.load(str(hom_resc_path))["M_homography"]

        _logger.info(f"Loaded original homography: {hom}")
        _logger.info(f"Loaded rescaled homography: {hom_resc}")

    else:
        fp_paths = sorted(list(Path(homography_dir).glob("*.csv")))
        if not fp_paths:
            raise FileNotFoundError(
                "No files in .csv format in the featurepoint directory!"
            )

        try:
            rgb_csv_path = [s for s in fp_paths
                            if "RGB" in str(s) or "rgb" in str(s)][0]
            tir_csv_path = [s for s in fp_paths
                            if "TIR" in str(s) or "tir" in str(s)][0]
        except IndexError as e:
            raise IndexError(
                "Invalid .csv file naming: No feature-point files found "
                "for both RGB and TIR in the feature-point directory!"
            ) from e

        # Load source points from csv path
        with open(rgb_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            RGB_pts = np.array(list(reader)).astype(np.float32)
        # Load destination points from csv path
        with open(tir_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            TIR_pts = np.array(list(reader)).astype(np.float32)

        # Rescale TIR (dest) points to image size according to rescaling factor
        # NOTE: This is a hardcoded rescaling for this model camera
        _logger.debug(f"TIR (destination) points pre-scaling: {TIR_pts}")
        TIR_pts_resc = (np.rint(TIR_pts * scale_factor))
        _logger.debug(f"TIR (destination) points post-scaling: {TIR_pts_resc}")

        if RGB_pts.shape != TIR_pts_resc.shape:
            raise ValueError(
                f"RGB/source ({RGB_pts.shape}) and TIR/destination "
                f"({TIR_pts_resc.shape}) shapes must match."
            )

        # Calculate original and rescaled Homography matrices
        hom, _ = cv2.findHomography(RGB_pts, TIR_pts)
        hom_resc, _ = cv2.findHomography(RGB_pts, TIR_pts_resc)
        _logger.info(f"Calculated original homography: {hom}")
        _logger.info(f"Calculated rescaled homography: {hom_resc}")

        # Save homography matrices to predefined path for future use
        np.savez(hom_path, M_homography=hom)
        np.savez(hom_resc_path, M_homography=hom_resc)

    return hom, hom_resc


def align_image(tir_stem: str, rgb_stem: str,
                M_hom: np.array, tir: np.array, rgb: np.array,
                out_dir: Path, no_details: bool):
    """
    Align RGB images to TIRs using calculated homography.

    :param tir_stem: (Str) TIR image name without suffix
    :param rgb_stem: (Str) RGB image name without suffix
    :param M_hom: Homography matrix
    :param tir: (Ndarray) TIR image 3d array (3 channels)
    :param rgb: (Ndarray) RGB image 3d array (3 channels)
    :param out_dir: destination directory
    :param no_details: (bool) if True, no detailed overlays to be saved
    """
    rows, cols, _ = np.shape(tir)

    # using previously determined transformation matrix
    aligned_rgb = cv2.warpPerspective(rgb, M_hom, (cols, rows))

    if no_details:
        # save only the aligned RGB
        cv2.imwrite(str(Path(out_dir, f"{rgb_stem}_aligned.jpg")), aligned_rgb)
        _logger.debug(f"aligned and saved ...{str(out_dir.stem)}")
    else:
        # overlay RGB and TIR images
        combined_image = cv2.addWeighted(tir, 0.25, aligned_rgb, 0.4, 0)

        cv2.imwrite(str(out_dir / f"{tir_stem}.jpg"), tir[:, :, 0])
        cv2.imwrite(str(out_dir / f"{rgb_stem}_aligned.jpg"), aligned_rgb)
        cv2.imwrite(str(out_dir / f"{tir_stem}_overlaid_RGB_TIR.jpg"),
                    combined_image)

        _logger.debug(f"aligned and saved ...{str(out_dir.stem)} "
                      f"with overlaying details")

    return aligned_rgb
