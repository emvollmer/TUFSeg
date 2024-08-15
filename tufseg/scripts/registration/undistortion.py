##!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import numpy as np
import cv2

'''
Helper functions for STEP 1 in the image preprocessing procedure:
Undistort RGB images using the given calibration file.
The calibration file includes the camera's intrinsic matrix and
distortion coefficients. It is calculated using the script in utils.

Aligning and merging the images comes in steps 2 and 3.
'''
_logger = logging.getLogger(__name__)


def undistort_image(img_stem: str,
                    img: np.array,
                    intrin_mat: np.ndarray,
                    dist_coeff: np.ndarray,
                    out_dir: Path
                    ):
    """
    Remove distortion from a single RGB image

    Args:
        img_stem: image name
        img: image
        intrin_mat: intrinsic matrix from calibration data
        dist_coeff: distortion coefficients from calibration data
        out_dir: destination directory for undistorted images

    Returns: undistorted image

    """
    filename = img_stem
    _logger.debug(f"Processing {filename}")

    # Warp source image to destination based on homography
    out_img = cv2.undistort(
        img,
        intrin_mat,
        dist_coeff,
        None,
    )

    cv2.imwrite(str(out_dir / f"{filename}_undistorted.jpg"), out_img)
    _logger.debug(f"undistorted and saved ...{str(out_dir.stem)}")

    return out_img
