#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-14 10:00:57
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-24 15:47:05
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch
import numpy
import random

from typing import Any, Literal, Iterable


def shuffle_sequence(sequence: Iterable) -> Iterable:
    indices = list(range(len(sequence)))
    random.shuffle(indices)
    shuffled_sequence = ( sequence[index] for index in indices )
    return shuffled_sequence


def make_reproducible(seed: int = 3407, mode: bool = True):
    assert 0 < seed, 'Seed must > 0 .'

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode)


def get_logging_metrics_str(metrics: dict[str, str]) -> str:
    metrics_str = ' '.join([f'[{metric_name}]={metric_value}' for metric_name, metric_value in metrics.items()])
    return metrics_str


def get_model_parameters_number(model: torch.nn.Module) -> int:
    parameters_number = dict()
    for name, parameters in model.named_parameters():
        root_name = name.split('.')[0]
        if root_name in parameters_number:
            parameters_number[root_name] += parameters.numel()
        else:
            parameters_number[root_name] = parameters.numel()

    return parameters_number


def get_device_descriptor(device: Literal['CPU', 'GPU'], index: int) -> torch.device:
    if device == 'CPU':
        device_name = 'cpu'

    if device == 'GPU':
        device_name = f'cuda:{index}'

    return torch.device(device_name)


def average_model_state_dict(model_state_dicts: list[dict], weights: list[float] | None = None) -> dict:
    if weights is None:
        weights = [1.0 / len(model_state_dicts)] * len(model_state_dicts)
    else:
        assert len(model_state_dicts) == len(weights)

    final_state_dict = dict()
    for model_state_dict, weight in zip(model_state_dicts, weights):
        for key, value in model_state_dict.items():
            if key in final_state_dict:
                final_state_dict[key] += value * weight
            else:
                final_state_dict[key] = value * weight
    return final_state_dict


# def save_operator_embedding(save_dirpath: pathlib.Path, weights: NDArray, op_dict: dict[str, int]):
#     save_dirpath.mkdir(parents=True, exist_ok=False)
#     weights_filepath = save_dirpath.joinpath(f'weights.npy')
#     op_dict_filepath = save_dirpath.joinpath(f'op_dict.json')
#     numpy.save(weights_filepath, weights)
#     save_json(op_dict, op_dict_filepath, indent=2)


# def load_operator_embedding(load_dirpath: pathlib.Path):
#     weights_filepath = load_dirpath.joinpath(f'weights.npy')
#     op_dict_filepath = load_dirpath.joinpath(f'op_dict.json')
#     weights = numpy.load(weights_filepath)
#     op_dict = load_json(op_dict_filepath)
#     return weights, op_dict