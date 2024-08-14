# Creating RGBT datasets from raw RGB and TIR images

This helper script is only necessary when starting off with raw RGB and TIR imagery that require concatenation before setup and model training can be performed. Registration (alignment) of the raw imagery is performed in three steps:
1. Undistorting RGBs
   - via camera calibration (by calculating intrinsic matrix and distortion coefficients)
   - uses `resources/calibration/calibration_data.npz`
2. Aligning RGBs with TIRs 
   - via homography matrix, calculated from manually determined matching feature points
   - uses `resources/alignment/homography_matrix_...npz` or - if non-existent - `/featurepoints_.
     ..csv`
3. Merging the results

With the default settings, this is done to generate outputs with:
 - 640x512 and 3750x3000 image resolution
 - 4 (R+G+B+T) and 2 (grayRGB+T) image channels

### Preparing a directory tree for data

If it doesn't exist yet, a directory tree needs to be generated in a working directory
to which the raw data can be stored and all results saved.
This can be accomplished by running the `generate_dirtree.py` script in one of two ways:

```
python3 -m scripts.registration.generate_dirtree --sets-dir path/to/dataset(s) --work-dir WORK_DIR
python3 -m scripts.registration.generate_dirtree --sets-list KA_01 KA_02 --work-dir WORK_DIR
```

The tree will be initialised depending on the names of the datasets.
These can either be provided using `set-dirs`, a path to the directory containing all dataset folders,
or `sets-list`, by listing the names themselves.

Additionally, channels and sizes can be set in case the default values require changing.
The following options exist:
- `channels`: `2ch`, `4ch` or the default `(2ch, 4ch)`,
- `sizes`: The inputs need to have the same aspect ratio and only two are permitted -
the original size of the thermal images and the goal (upscaled) format.
The default here is `(640x512, 3750x3000)`.

This results in the following directory tree:
```
WORK_DIR
├── raw
│   └── images
│       ├── DS_01   ---> dataset 1, i.e. KA_01 (Karlsruhe 1)
│       │   ├── RGB      ------> folder for raw RGB images to be stored in
│       │   └── Thermal  ------> folder for raw TIR images to be stored in
│       └── ...
└── merged
```
To run the registration script, you need to place your raw imagery into the directory as shown.

### Executing the RGB and TIR registration

All three steps are performed using the `register_RGB_TIR.py` script.
It can be called using all default values as follows:

```
python3 -m scripts.registration.register_RGB_TIR --work-dir WORK_DIR -n 5
```
with the working directory the same as above and the number of jobs (`-n`) as many as you want to run in parallel.

The required inputs are defaulted to be the `resources/` directory. These include:

- `calibration_path`: Path to correction calibration file, default: `resources/calibration/calibration_data.npz`
- `homography_dir`: Folder containing homography matrices and/or CSV featurepoint files, default: `resources/alignment/`

> **NOTE:** The default channel count `channels` and image size formats `sizes` can be changed.
> Please ensure they match the directory tree though!

A more complex call with changed defaults could look as follows:

```
python3 -m scripts.registration.register_RGB_TIR --work-dir WORK_DIR --calibration-path CALIB_PATH --homography-dir HOM_DIR
--channels 2ch --sizes 640x512 5000x4000 -nd -n 5 -vv
```
Use the `--help` flag for more information.

> **NOTE:** Depending on the available computational memory, number of parallel jobs,
> and other selected flags (i.e. (non-)usage of `-nd`), the registration process can take quite some time!

### Resulting directory tree

With the example default command, running the code will result in the following directory tree:
```
WORK_DIR
├── raw
│   └── images
│       ├── DS_01   ---> dataset 1, i.e. KA_01 (Karlsruhe 1)
│       │   ├── RGB             ------> raw RGB images, i.e. DJI_0_0002.jpg
│       │   ├── RGB_aligned     ------> aligned RGB images
│       │   │   ├── 3750x3000      ------> i.e. DJI_0_0001_R.jpg, DJI_0_0001_R_overlaid_RGB_TIR.jpg, DJI_0_0002_aligned.jpg
│       │   │   └── 640x512        ------> -"-
│       │   ├── RGB_undistorted ------> undistorted RGB images, i.e. DJI_0_0002_undistorted.jpg
│       │   └── Thermal         ------> raw TIR images, i.e. DJI_0_0001_R.JPG
│       └── ...
└── merged
    ├── 2ch
    │   ├── 3750x3000
    │   │   └── images  ---> merged images
    │   │       ├── DS_01   ---> dataset 1, i.e. KA_01 (Karlsruhe 1)
    │   │       │   ├── DJI_0_0001_R.npy
    │   │       │   └── ...
    │   │       └── ...
    │   └── 640x512
    │       └── images  ---> merged images
    │           └── ...
    └── 4ch
        ├── 3750x3000
        └── 640x512
```
The data currently utilised in the subsequent [setup](../setup/README.md) and [model training](../segm_models/README.md) is from `4ch/3750x3000`. 
This can be used as the image base directory for the [setup scripts](../setup/README.md), though it 
requires the addition of the necessary JSON annotations.