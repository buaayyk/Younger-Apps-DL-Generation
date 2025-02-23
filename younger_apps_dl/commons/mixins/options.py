#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-16 14:13:12
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-23 16:29:37
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Type, TypeVar, Generic
from pydantic import BaseModel, ValidationError


OptionsType = TypeVar('T', bound=BaseModel)

class OptionsMixin(Generic[OptionsType]):
    _options_: Type[OptionsType] = None

    def __init__(self, **options):
        if self.__class__._options_ is None:
            raise NotImplementedError(f'Classes with this `OptionsMixin` must implement a corresponding configuration class and define it as a class attribute.')

        try:
            self._options = self.__class__._options_(**options)
        except ValidationError as exception:
            raise ValueError(f"Invalid `options` for {self.__class__.__name__}: {exception}")
        super().__init__()

    @property
    def options(self) -> OptionsType:
        return self._options
