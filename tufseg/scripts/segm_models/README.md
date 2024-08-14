# Creating, evaluating, and using `segmentation_model` U-Net models

All functions can be called from the command line as modules. Check out which parameters you need for running them by doing:
```
python -m path.to.script --help
```

In all functions, use the flags `-v` (verbose) or `-vv` (very verbose) for more logging information. The default is `-quiet` which is logging on WARNING mode.

### Function calls

Example function calls from the terminal are demonstrated below

#### For training

```
python -m tufseg.scripts.segm_models.train_UNet -dst /path/to/model_unet/outputs/ --channels 3 --processing basic --cfg-options epochs=8
```

- The flags `-src /.../datasets/ -split /.../model_unet/data/` are optional. If the setup scripts were previously executed, you don't have to define these -  the defaults will taken from the config.

- The flag `--channels` can be set to either `3` or `4` for adapted 3 channel or using the data as it is. Default is `4`.

- The flag `--processing` can be set to either `basic` for the data as is, `vignetting` for the removal of vignetting in the thermal channel or `retinex_unsharp` for filtered inputs. Last of these only works in combination with `channels=3`. Default is `basic`.

- An additional flag `--only-tir` can optionally be set if only the thermal image channels are to be used for training, thus ignoring the RGB information.

- The flag `--cfg-options` can be used to change the hyperparameters seed number, epoch count, batch size and learning rate from the defaults to user specific values. Defaults are: `cfg = {'seed': 1000, 'lr': 0.001, 'epochs': 3, 'batch_size': 8}`.

#### For evaluation

```
python -m tufseg.scripts.segm_models.evaluate_UNet /path/to/model_unet/outputs/2023-08-16-133200/
```
The required path is the path to the model timestamp folder. Optional flags are `-src /.../datasets/` and `-split /.../model_unet/data/`. By default, the values from the training config will be used (saved to `run_config.json` in the model folder).

#### For inference / prediction

```
python -m tufseg.scripts.segm_models.infer_UNet -img /path/to/DJI_0_0001_R.npy -model /path/to/model_unet/outputs/2023-08-16_18-05-23/ -save
```

- The flag `-mask` can be used if a ground truth segmentation mask exists and should be plotted side by side to the prediction. If such a file exists in the `-img` folder structure, it is used by default.

- The flag `-save` saves the results into a subfolder `predictions` in the `--model-dir`. Visualisations include in a side-by-side overview of RGB, T and prediciton mask as well as the predictions on their own.

- Another flag, `display`, allows plotting straight to the console.


### Monitoring energy consumption with perun

To monitor energy consumption using [perun](https://github.com/Helmholtz-AI-Energy/perun), execute training via the bash script `train.sh` as follows:

```
. tufseg/scripts/segm_models/train.sh -dst /path/to/model_unet/outputs/ --channels 3 --processing vignetting --only-tir --cfg-options epochs=3 --quiet
```

- The flags `--channels 3 --processing vignetting --only-tir` are optional. Without them, the defaults from the config will be chosen: `channels=4`, `processing=basic`, `only_tir=False`

- You can optionally add in `-src /.../datasets/ -split /.../model_unet/data/`, but these values will automatically be taken from the config created during the setup module calls (`setup.sh`).

What this does is call the train script as follows:
```
perun monitor --format csv -m tufseg.scripts.segm_models.train_UNet -dst /path/to/model_unet/outputs/ --channels 3 --processing vignetting --only-tir --cfg-options epochs=3
```
and then copies the perun results into the timestamp model directory that was created during the model run. As usual, the logging results in INFO mode are saved to the same model directory.

After training, the model is automatically evaluated via the below command and using that same timestamp folder as the model directory:
```
python -m tufseg.scripts.segm_models.evaluate_UNet /path/to/model_unet/outputs/timestamp/
```

> **NOTE:**  
> This works with adapted perun package, which is automatically installed via the 
> requirements file. Otherwise, it can be installed with:
> ```
> pip install "git+https://github.com/emvollmer/perun.git"
> ```
