#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-16 14:13:12
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-08 17:56:29
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Type, TypeVar, ClassVar, Generic
from pydantic import BaseModel


OPTIONS_TYPE = TypeVar('OPTIONS_TYPE', bound=BaseModel)

class OptionsMixin(Generic[OPTIONS_TYPE]):
    OPTIONS: ClassVar[Type[OPTIONS_TYPE]]

    def __init__(self, options: OPTIONS_TYPE):
        if self.__class__.OPTIONS is None:
            raise NotImplementedError(f'Classes with this `OptionsMixin` must implement a corresponding OPTIONS class and define it as a class attribute.')

        if not isinstance(options, self.__class__.OPTIONS):
            raise TypeError(f"Expected options of type {self.__class__.OPTIONS.__name__}, but got {type(options).__name__}")

        self._options_ = options
        super().__init__()

    @property
    def options(self) -> OPTIONS_TYPE:
        return self._options_
