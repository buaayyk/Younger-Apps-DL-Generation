#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-14 22:48:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-21 13:25:16
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from abc import ABC, abstractmethod
from typing import Literal, Type

from younger_apps_dl.commons.mixins.options import OptionsMixin


class BaseComponent(OptionsMixin, ABC):
    def __init__(self, configuration: dict) -> None:
        super().__init__(configuration)


COMPONENT_REGISTRY: dict[ Literal['model', 'dataset'], dict[ str, Type[BaseComponent] ] ] = dict(
    model = dict(),
    dataset = dict(),
)


def register_component(
    kind: Literal['model', 'dataset'],
    name: str,
):
    assert kind in {'model', 'dataset'}
    assert name not in COMPONENT_REGISTRY[kind]
    def wrapper(cls: Type[BaseComponent]) -> Type[BaseComponent]:
        COMPONENT_REGISTRY[kind][name] = cls
        return cls
    return wrapper

from .models import *
from .datasets import *
