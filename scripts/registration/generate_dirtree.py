#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast
import argparse
from tqdm import tqdm
from pathlib import Path

"""
Generate a directory tree which can be populated with a raw dataset
and be used to save results from preprocessing said dataset. This can
be done either for a list of datasets or a single one (for example if
a single dataset should be added to already completed preprocessing)
"""


def get_args():
    parser = argparse.ArgumentParser(
        description='''Generate a directory tree for raw datasets and merging results.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sets-dir', type=Path, help="Directory containing folders for all datasets",
                       metavar="SETS_DIR", dest='sets_dir')
    group.add_argument('--sets-list', nargs='+', help="List containing strings of all dataset names",
                       metavar="SETS_LIST", dest='sets_list')
    parser.add_argument('-work', '--work-dir', type=Path, required=True,
                        help="Directory to which raw and merged data should be saved", metavar="WORK_DIR",
                        dest='work_dir')
    parser.add_argument('-channels', required=False, default="['2ch', '4ch']", const="['2ch', '4ch']", nargs='?',
                        choices=["2ch", "4ch", "['2ch', '4ch']"], metavar="CHANNELS", dest='channels_str',
                        help="Choice of different channels of resulting numpy files, "
                             "with 4ch = RGBT, 2ch = greyRGB + T")
    parser.add_argument('-sizes', nargs=2, required=False, default=['640x512', '3750x3000'],
                        help="Two size format options for results. List the original TIR size first, "
                             "then the destination size for re-/upscaling.",
                        metavar="SIZES", dest='sizes_list')
    return parser.parse_args()


def main(
    sets_dir,
    sets_list,
    channels_str: str,
    work_dir: Path,
    sizes_list: list
):

    if sets_dir is not None:
        assert Path(sets_dir).exists(), \
            "Provided sets-dir does not exist! Please provide an existing path."

        set_names = [i.stem for i in sorted(list(sets_dir.glob("[!.]*")))]

    else:
        set_names = sets_list

    try:
        channels = ast.literal_eval(channels_str)
    except SyntaxError:
        channels = [channels_str]

    generate_dirtree(
        dataset_names=set_names,
        base_dir=work_dir,
        img_channels=channels,
        img_sizes=sizes_list
    )


def generate_dirtree(
        dataset_names: list,
        base_dir: Path,
        img_channels=['2ch', '4ch'],
        img_sizes=['640x512', '3750x3000'],
):
    assert img_sizes, \
        "Please provide desired image size in list form, i.e. ['640x512', '3750x3000']"

    for dataset_name in tqdm(dataset_names):
        generate_dataset_dirtree(dataset_name, base_dir, img_channels, img_sizes)


def generate_dataset_dirtree(
        dataset_name: str,
        work_dir: Path,
        img_channels=['2ch', '4ch'],
        img_sizes=['640x512', '3750x3000'],
):
    directory_list = [
        Path(work_dir, "raw", "images", dataset_name, "RGB"),
        Path(work_dir, "raw", "images", dataset_name, "Thermal"),
        Path(work_dir, "raw", "images", dataset_name, "RGB_undistorted")
    ]

    for s in img_sizes:
        # # uncomment to only add a folder tree if multiple different image sizes are provided
        # if len(img_sizes) == 1:
        #     s = ""

        directory_list.append(
            Path(work_dir, "raw", "images", dataset_name, "RGB_aligned", s)
        )
        for ch in img_channels:
            # # uncomment to only add a folder tree if multiple different image channel options are provided
            # if len(img_channels) == 1:
            #     ch = ""

            directory_list.append(
                Path(work_dir, "merged", ch, s, "images", dataset_name)
            )

    # create directories
    for directory in directory_list:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = get_args()
    main(
        args.sets_dir,
        args.sets_list,
        args.channels_str,
        args.work_dir,
        args.sizes_list
    )
