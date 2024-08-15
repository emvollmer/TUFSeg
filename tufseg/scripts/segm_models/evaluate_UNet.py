#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate trained models
"""
# import built-in dependencies
import argparse
import json
import logging
import os
from pathlib import Path
import warnings

# suppress messages from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning,
                        message="y_pred contains classes not in y_true")

# import external dependencies
import numpy as np
from sklearn import metrics as skmetrics
import tensorflow as tf
from tqdm import tqdm

from tufseg.scripts.segm_models._utils import (
    configure_logging, ModelLoader, ImageProcessor, MaskProcessor
)
from tufseg.scripts.configuration import read_conf

logging.getLogger('h5py').setLevel(logging.ERROR)
# --------------------------------------
_logger = logging.getLogger("evaluate")

config: dict


def parse_args():
    """Get user required arguments"""
    parser = argparse.ArgumentParser(description='Evaluating model')
    parser.add_argument('model_root', metavar='MODEL_ROOT',
                        help='Path to destination directory that contains '
                             'model and model configs.')

    # Combine log level options into one with a default value
    log_levels_group = parser.add_mutually_exclusive_group()
    log_levels_group.add_argument('--quiet', dest='log_level',
                                  action='store_const', const=logging.WARNING,
                                  help='Show only warnings.')
    log_levels_group.add_argument('-v', '--verbose', dest='log_level',
                                  action='store_const', const=logging.INFO,
                                  help='Show verbose log messages (info).')
    log_levels_group.add_argument('-vv', '--very-verbose', dest='log_level',
                                  action='store_const', const=logging.DEBUG,
                                  help='Show detailed log messages (debug).')
    log_levels_group.set_defaults(log_level=logging.INFO)
    return parser.parse_args()


def main(
        model_root: Path,
        log_level=logging.WARNING
):
    """
    Coordinate model evaluation with user inputs
    and save evaluation results to a .json in the model directory
    """
    global config
    config = read_conf(Path(model_root, "run_config.json"))
    configure_logging(_logger, log_level)

    # log user provided inputs
    main_args = locals()
    for k, v in main_args.items():
        _logger.info(f'Parsed user argument - {k}:\t{v}')

    # load model
    model_loader = ModelLoader(config, model_root)
    model = model_loader.model
    NUM_CLASSES = model_loader.NUM_CLASSES

    # load data (y_test NOT onehot encoded due to sigmoid activation)
    img_proc = ImageProcessor(config)
    X_test = img_proc.process_images(root="test", overwrite=False)

    mask_proc = MaskProcessor(config)
    y_test = mask_proc.load_masks(root="test", overwrite=False)

    # evaluate
    _logger.info("Evaluate model on TEST dataset...")
    sm_mean_scores = evaluate_sm(
        X_test, y_test, model=model, NUM_CLASSES=NUM_CLASSES
    )
    combined_mean_scores, combined_class_mean_scores = (
        evaluate_sklearn_imagewise(
            X_test, y_test, model=model
        )
    )
    holistic_mean_scores = evaluate_sklearn_holistic(
        X_test, y_test, model=model
    )

    # save results
    eval_json_path = Path(model_root, "eval.json")
    _logger.info(f"Saving evaluation scores to '{eval_json_path}'...")
    with open(eval_json_path, "w") as json_file:
        json.dump(
            {
                "sm metrics": sm_mean_scores,
                "sklearn metrics - holistic results": holistic_mean_scores,
                "sklearn metrics - combined imagewise results":
                    combined_mean_scores,
                "sklearn metrics - combined imagewise classwise results":
                    combined_class_mean_scores,
            },
            json_file, indent=4
        )


def evaluate_sm(X_test, y_test, model, NUM_CLASSES):
    """
    Evaluate trained model using test dataset
    on select segmentation_model toolkit metrics.

    :param X_test: test images
    :param y_test: test masks
    :param model: loaded model
    :param NUM_CLASSES: number of classes
    :return: mean_scores: (dict) evaluation metrics applied to model
    """
    _logger.info("Evaluating model - segmentation_model metrics.")

    # Evaluate the model on the test dataset
    y_test_onehot = np.asarray(tf.one_hot(y_test, NUM_CLASSES))
    results = model.evaluate(
        X_test, y_test_onehot, verbose=1
    )

    # Determine the order of evaluation metrics
    metrics_names = model.metrics_names

    # Extract the evaluation metrics from the results based on their names
    mean_scores = dict(zip(metrics_names, results))

    print("Evaluation results (segmentation_model metrics):")
    for key, val in mean_scores.items():
        print(
            f"{key.ljust(max(len(k) for k in mean_scores.keys()))}: "
            f"{round(val, 4)}"
        )

    return mean_scores


def evaluate_sklearn_imagewise(X_test, y_test, model):
    """
    Evaluate trained model using test dataset by inferring on images
    one at a time, calculating select scikit-learn metrics for each
    and combining the results to average scores for the whole test dataset.

    :param X_test: test images
    :param y_test: test masks
    :param model: loaded model
    :return: mean_scores: (dict) evaluation metrics applied to model;
             mean_classwise_scores: (dict) class-wise evaluation metrics
             (excluding background)
    """
    _logger.info(
        "Evaluate model by evaluating test images individually and "
        "combining scores."
    )

    sklearn_metrics = config['eval']['SKLEARN_METRICS']
    total_scores = {metric: [] for metric in sklearn_metrics.keys()}

    classes = config['data']['masks']['labels']
    classes.insert(0, "background")
    classwise_scores = {metric: {label: [] for label in classes}
                        for metric in sklearn_metrics.keys()}

    for test_img, test_mask in tqdm(zip(X_test, y_test)):
        # Prediction
        prediction = model.predict(
            np.expand_dims(test_img, axis=0),
            verbose=0
        )
        class_predictions = np.argmax(prediction, axis=-1)
        class_predictions = np.squeeze(class_predictions)

        # Evaluate each metric with scikit learn functions and their params
        for metric_name, metric_attrib in sklearn_metrics.items():
            function = getattr(skmetrics, metric_attrib['func'])
            metric_params = metric_attrib["params"]

            score = function(
                test_mask.flatten(),
                class_predictions.flatten(),
                **metric_params
            )
            total_scores[metric_name].append(score)

            # skip weighted and balanced functions for class-wise evaluation
            if (
                    'weighted' in metric_name.lower()
                    or 'balanced' in metric_name.lower()
            ):
                continue

            # evaluate class-wise
            for class_id, class_name in enumerate(classes):
                class_mask_true = (test_mask == class_id)
                class_mask_pred = (class_predictions == class_id)

                # evaluate only images that actually contain the class
                if np.any(class_mask_true) or np.any(class_mask_pred):
                    metric_singleclass_params = metric_params.copy()
                    if 'average' in metric_params.keys():
                        metric_singleclass_params['average'] = 'binary'

                    class_score = function(
                        class_mask_true.flatten(),
                        class_mask_pred.flatten(),
                        **metric_singleclass_params
                    )
                    classwise_scores[metric_name][class_name].append(
                        class_score
                    )

    mean_scores = {f"mean_{k}": np.mean(v) for k, v in total_scores.items()}

    print("Combined image-wise evaluation results (scikit-learn metrics):")
    for key, val in mean_scores.items():
        print(
            f"{key.ljust(max(len(k) for k in mean_scores.keys()))}: "
            f"{round(val, 4)}"
        )

    mean_classwise_scores = {}
    for metric_name in sklearn_metrics.keys():
        if (
                'weighted' not in metric_name.lower()
                and 'balanced' not in metric_name.lower()
        ):
            classwise_metric_name = f'mean_{metric_name}_classwise'
            classwise_scores_dict = {
                class_name: np.mean(scores)
                for class_name, scores in classwise_scores[metric_name].items()
            }
            mean_classwise_scores[classwise_metric_name] = (
                classwise_scores_dict
            )

    print("------------------- class-wise evaluation results:")
    for key1, val1 in mean_classwise_scores.items():
        print(f"{key1}")
        for key2, val2 in val1.items():
            print(
                f"{key2.ljust(max(len(k) for k in val1.keys()))}: "
                f"{round(val2, 4)}"
            )

    return mean_scores, mean_classwise_scores


def evaluate_sklearn_holistic(X_test, y_test, model):
    """
    Evaluate trained model using test dataset
    by inferring on all data at once and evaluating with
    select scikit-learn metrics.

    :param X_test: test images
    :param y_test: test masks
    :param model: loaded model
    :return: mean_scores: (dict) evaluation metrics applied to model
    """
    _logger.info(
        "Evaluate model by evaluating all images together and "
        "calculating scores."
    )

    mean_scores = {}
    for name, attrib in config['eval']['SKLEARN_METRICS'].items():
        # predict over the entire dataset
        all_predictions = model.predict(X_test)
        all_class_predictions = np.argmax(all_predictions, axis=-1)

        # evaluate each metric with scikit learn functions and their params
        function = getattr(skmetrics, attrib['func'])
        metric_params = attrib["params"]

        mean_score = function(
            y_test.flatten(),
            all_class_predictions.flatten(),
            **metric_params
        )
        mean_scores[f"mean_{name}"] = mean_score

    print("Holistic evaluation results (scikit-learn metrics):")
    for key, val in mean_scores.items():
        print(
            f"{key.ljust(max(len(k) for k in mean_scores.keys()))}: "
            f"{round(val, 4)}"
        )

    return mean_scores


if __name__ == "__main__":
    args = parse_args()
    main(model_root=args.model_root,
         log_level=args.log_level)
