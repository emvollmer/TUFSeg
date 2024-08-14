##!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import cv2
import numpy as np
from pathlib import Path

'''
Helper functions for STEP 3 in the image preprocessing procedure:
Merging processed RGBs and TIRs to numpy files.
'''
_logger = logging.getLogger(__name__)


def merge_images(tir: np.array, rgb: np.array, work_dir: Path, channels: list, end_of_path: Path):
    """
    Merge aligned RGB and TIR to both 4 channel (R + G + B + T)
    and 2 channel (grayscale RGB + T) npy arrays.

    Args:
        tir: (Ndarray) TIR image 3d array (3 channels)
        rgb: (Ndarray) RGB image 3d array (3 channels)
        work_dir: (Path) working directory
        channels: (list) list of either ['2ch'], ['4ch'] or ['2ch', '4ch']
        end_of_path: (Path) rest of path for specific numpy file
    """

    for channel in channels:
        out_path = Path(work_dir, "merged", channel, end_of_path)

        if channel == "2ch":
            # merge grayscale RGB and TIR to a 2ch numpy array
            rgb_gray = np.expand_dims(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), axis=2)
            out_2ch = np.concatenate(
                (rgb_gray, tir),
                axis=2
            )
            # save using thermal layer's name as they were used to create the annotations
            np.save(str(out_path), out_2ch)
            _logger.debug(f"Merged 2 channel file saved to work directory ...{str(Path(channel, end_of_path))}")
        else:
            # merge all layers to 4ch numpy array
            out_4ch = np.concatenate(
                (rgb, tir),
                axis=2
            )
            # save using thermal layer's name as they were used to create the annotations
            np.save(str(out_path), out_4ch)
            _logger.debug(f"Merged 4 channel file saved to work directory ...{str(Path(channel, end_of_path))}")

    return
