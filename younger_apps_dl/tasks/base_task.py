#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-09 22:36:51
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-13 22:30:02
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch
import torch.utils.data

from typing import Any, Callable, Literal
from collections import OrderedDict

from younger.commons.logging import set_logger, Logger
from younger.commons.constants import YoungerHandle
from younger_apps_dl.utils.neural_network import get_device_descriptor


class YoungerAppsDLBaseTask(object):
    def __init__(self, custom_config: dict) -> None:
        custom_logging_config = custom_config.get('logging', dict())

        logging_config = dict()
        logging_config['name'] = custom_logging_config.get('name', YoungerHandle.ApplicationsName + '-DL-Task-' + 'Base')
        logging_config['mode'] = custom_logging_config.get('mode', 'console')
        logging_config['level'] = custom_logging_config.get('level', 'INFO')
        logging_config['filepath'] = custom_logging_config.get('filepath', None)
        self._logging_config = logging_config

        self._device_descriptor = None
        self._logger = None

    @property
    def device_descriptor(self) -> torch.device:
        if self._device_descriptor is None:
            self._device_descriptor = get_device_descriptor('CPU', 0)
        return self._device_descriptor

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = set_logger(self._logging_config['name'], mode=self._logging_config['mode'], level=self._logging_config['level'], logging_filepath=self._logging_config['filepath'])
        return self.logger

    def to(self, device_descriptor: torch.device):
        self._device_descriptor = device_descriptor

    def reset(self):
        self._device_descriptor = None
        self._logger = None

    def update_learning_rate(self, stage: Literal['Step', 'Epoch'], **kwargs):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        return

    @property
    def model(self) -> torch.nn.Module:
        raise NotImplementedError

    @model.setter
    def model(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @property
    def train_dataset(self):
        raise NotImplementedError

    @property
    def valid_dataset(self):
        raise NotImplementedError

    @property
    def test_dataset(self):
        raise NotImplementedError

    def train(self, minibatch: Any) -> tuple[torch.Tensor, OrderedDict[str, tuple[torch.Tensor, Callable | None]]]:
        raise NotImplementedError

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict[str, tuple[torch.Tensor, Callable | None]]:
        raise NotImplementedError

    def prepare_cli(self):
        self.logger.warning(f'Not Implemented!')

    def cli(self, **kwargs):
        self.logger.warning(f'Not Implemented!')

    def api(self, **kwargs):
        self.logger.warning(f'Not Implemented!')
