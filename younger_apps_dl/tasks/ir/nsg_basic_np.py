#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-09 22:36:51
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-23 22:54:09
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import abc
import pydantic
import torch
import torch.utils.data

from typing import Any, Literal, Callable

from younger.commons.logging import set_logger, Logger
from younger.commons.constants import YoungerHandle

from younger_apps_dl.commons.utils import get_device_descriptor


class BaseTask(object):
    def __init__(self, configuration: dict[str, Any]) -> None:
        self._configuration = configuration

        logging_config = self._configuration.get('logging', dict())
        self._logging_config = dict()
        self._logging_config['name'] = logging_config.get('name', YoungerHandle.ApplicationsName + '-DL-Task-' + 'Base')
        self._logging_config['mode'] = logging_config.get('mode', 'console')
        self._logging_config['level'] = logging_config.get('level', 'INFO')
        self._logging_config['filepath'] = logging_config.get('filepath', None)

    def initialize(self):
        self._logger: Logger | None = None
        self._device: torch.device | None = None

        self._epoch: int | None = None
        self._step: int | None = None

        self._model: torch.nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None

        self._training_dataset = None
        self._validation_dataset = None
        self._test_dataset = None

    @property
    def configuration(self) -> dict[str, Any]:
        return self._configuration

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = set_logger(self._logging_config['name'], mode=self._logging_config['mode'], level=self._logging_config['level'], logging_filepath=self._logging_config['filepath'])
        return self.logger

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = get_device_descriptor('CPU', 0)
        return self._device

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self.__class__.setup_model(self.configuration, self.logger)
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module) -> torch.nn.Module:
        self._model = model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = self.__class__.setup_optimizer(self.configuration, self.logger)
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        self._optimizer = optimizer

    @property
    def training_dataset(self):
        if self._training_dataset is None:
            self._training_dataset = self.__class__.setup_training_dataset(self.configuration, self.logger)
        return self._training_dataset

    @property
    def validation_dataset(self):
        if self._validation_dataset is None:
            self._validation_dataset = self.__class__.setup_validation_dataset(self.configuration, self.logger)
        return self._validation_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            self._test_dataset = self.__class__.setup_test_dataset(self.configuration, self.logger)
        return self._test_dataset

    def train(self, minibatch: Any) -> torch.Tensor:
        raise NotImplementedError

    def log_train(self, loss: torch.Tensor) -> dict[str, tuple[torch.Tensor, Callable | None]]:
        raise NotImplementedError

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def log_eval(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> dict[str, tuple[torch.Tensor, Callable | None]]:
        raise NotImplementedError

    def cli(self, **kwargs):
        self.logger.warning(f'Not Implemented!')

    def api(self, **kwargs):
        self.logger.warning(f'Not Implemented!')

    def to(self, device_descriptor: torch.device):
        self._device_descriptor = device_descriptor
        self.model.to(self._device_descriptor)

    def update_status(self, stage: Literal['Step', 'Epoch'], epoch: int, step: int, **kwargs):
        assert stage in {'Step', 'Epoch'}, f'Only Support \'Step\' or \'Epoch\''
        self._epoch = epoch
        self._step = step

    # Build Model
    logger.info(f'-> Preparing Model ...')

    # Print Model
    task.logger.info(f'-> Model Specs:')
    parameters_number = get_model_parameters_number(task.model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    task.logger.info(
        f'\n======= v Model Architecture v ======='
        f'\n{task.model}'
        f'\n'
        f'\n====== v Number of Parameters v ======'
        f'\n{parameters_number_str}'
    )


    task.logger.info(f'-> Preparing Datasets ...')

    task.logger.info(f'   Dataset Split Sizes:')
    task.logger.info(f'    * Train - {len(task.train_dataset)}')
    if task.valid_dataset:
        task.logger.info(f'    * Valid - {len(task.valid_dataset)}')
    else:
        task.logger.info(f'    * Valid - Not Provided')



    checkpoint = dict()
    checkpoint['Epoch'] = epoch
    checkpoint['Step'] = step
    checkpoint['model_state'] = task.model.module.state_dict() if is_distribution else task.model.state_dict()
    checkpoint['optimizer_state'] = task.optimizer.state_dict()
    save_checkpoint(checkpoint, checkpoint_path=checkpoint_dirpath, checkpoint_name=checkpoint_name, keep_number=keep_number)