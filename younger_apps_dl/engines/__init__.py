#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-15 15:17:39
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-08 18:46:20
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from abc import ABC, abstractmethod
from typing import Literal, Type

from younger_apps_dl.commons.mixins.options import OptionsMixin, OPTIONS_TYPE


class BaseEngine(OptionsMixin[OPTIONS_TYPE], ABC):
    def __init__(self, options: OPTIONS_TYPE):
        super().__init__(options)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


ENGINE_REGISTRY: dict[ Literal['trainer', 'evaluator', 'predictor', 'preprocessor', 'postprocessor'] , dict[ str, Type[BaseEngine] ] ] = dict(
    trainer = dict(),
    evaluator = dict(),
    predictor = dict(),
    preprocessor = dict(),
    postprocessor = dict(),
)


def register_engine(
    kind: Literal['trainer', 'evaluator', 'predictor', 'preprocessor', 'postprocessor'],
    name: str,
):
    assert kind in {'trainer', 'evaluator', 'predictor', 'preprocessor', 'postprocessor'}
    assert name not in ENGINE_REGISTRY[kind]
    def wrapper(cls):
        assert issubclass(cls, BaseEngine)
        ENGINE_REGISTRY[kind][name] = cls
        return cls
    return wrapper

from .preprocessors import *
from .postprocessors import *
from .trainers import *
from .evaluators import *
from .predictors import *
