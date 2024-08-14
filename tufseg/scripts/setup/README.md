# Setting up for `segmentation_model` toolbox use

All modules can be called from the command line. Check out which parameters you need for running them by doing:
```
python -m path.to.script --help
```

In all modules, use the flags `-v` (verbose) or `-vv` (very verbose) for more logging information. The default is `-quiet` which is logging on WARNING mode.

### Function calls

Example function calls from the terminal are described below.

1. For generating the segmentation masks from the JSON annotation files
```
python -m tufseg.scripts.setup.generate_segm_masks -j /.../datasets/annotations/jsons/ -i /.../datasets/images/ --save-for-view
```
The script saves the masks in .npy format, but by adding the `save-for-view` flag, they'll be saved in .png format to the same directory for reviewing the annotations.

2. For splitting the data (images and masks) into train and test datasets
```
python -m tufseg.scripts.setup.train_test_split -v
```
- The optional flag `-src /.../datasets/` can be set to define the data source. Without it, the config value (saved by `generate_segm_masks.py` is used.

- The optional flag `-dst /.../data/split/` can be set to define the destination directory. Only do this if you want the processed data for training to be saved individually again. Without it, the `test.txt` and `train.txt` files will be saved to the source data directory and used from there for training.

- The optional flag `--test-size` can be defined as a decimal to define the percentage of the 
  test set. Default is `0.2`.
 
- Additional flags `--no-anno-check`, `--no-class-check`, or `--no-set-check` can be added to define the split without ensuring some basic aspects, such as the representation of all categories and all datasets (cities) in both train and test sets.

### Setting up with a bash script

Alternatively to individual module calls, you can execute the whole process at once with the `setup.sh` bash script.

Simply run:
```
. tufseg/scripts/setup/setup.sh -i /.../datasets/images/ -j /.../datasets/annotations/jsons/ -v
```

To create the split data directory to which the preprocessed images themselves can be saved during training, run:

```
. scripts/setup/setup.sh -i /.../datasets/images/ -j /.../datasets/annotations/jsons/ -dst /.../data/split/ -v
```

Remember though, this will fill up your `-dst` with additional data you don't necessarily need.

- Additional optional flags for both calls are: `--save-for-view`, `--test-size 0.2`
