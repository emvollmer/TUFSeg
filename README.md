# Thermal Urban Feature Segmentation (TUFSeg)

This repository contains scripts for the semantic segmentation of thermal urban features
using the [segmentation_models toolbox](https://github.com/qubvel/segmentation_models).
The popular U-Net is used as the base and retrained on differently processed RGBT datasets with
masks generated from json annotation files.

## Installation

The code was tested with Python version `3.8`. Tensorflow and keras `2.10` are used for model
training.

The repository modules can be run directly via command line as described in the READMEs
throughout the repo. Alternatively, it is installable as a package and can be used f.e.
in combination with APIs. The installation procedures differ according to the
descriptions below.

> **NOTE**:
> The [Perun](https://github.com/Helmholtz-AI-Energy/perun) package is used for energy
> monitoring. However, for it to work properly in combination with module calls, [this forked
> and adapted repository](https://github.com/emvollmer/perun.git) is installed instead.

### For direct usage of the modules

To install all required packages, run:
```
pip install -r requirements.txt
```

### For use as a package

Install the source code with pip with:

```
pip install .
```

For editable installation, run:

```
pip install -e .
```

## Repository structure

```
.
├── LICENSE                 ---> license file
├── README.md               ---> this file
├── VERSION                 ---> version file
├── pyproject.toml          ---> configuration file for python project and installation
├── requirements-test.txt   ---> requirements file for testing
├── requirements.txt        ---> requirements file for library installation
├── tufseg
│   ├── __init__.py
│   ├── resources
│   │   ├── alignment           ---> resources used in raw image registration for alignment
│   │   └── calibration         ---> resources used in raw image registration for camera calibration
│   └── scripts 
│       ├── __init__.py
│       ├── configuration.py    ---> configuration used across setup and segm_models
│       ├── registration        ---> optional scripts for merging raw RGB and thermal imagery into RGBTs
│       ├── segm_models         ---> required scripts for training, evaluating, and inferring with segmentation model(s)
│       └── setup		---> required scripts for RGBT processing for model training
├── tests		   ---> pytest tests
└── tox.ini		   ---> configuration file for tox tests
```

## License

This project is released under the [BSD-3-Clause license](https://github.com/emvollmer/TUFSeg/blob/main/LICENSE).

## Acknowledgement

The authors acknowledge support by the state of Baden-Württemberg through bwHPC. This work is funded by European Union through the AI4EOSC project (Horizon Europe) under Grant number 101058593.
We thank Marinus Vogl and the Air Bavarian GmbH for their support with equipment and service for the recording of images.
