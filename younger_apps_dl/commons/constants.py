#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-31 21:39:47
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-03 16:42:21
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from younger.commons.constants import Constant


class YADL_DATASET_ADDRESS(Constant):
    def initialize(self) -> None:
        self.ENDPOINT = 'https://younger.yangs.ai/public/'

YADL_Dataset_Address = YADL_DATASET_ADDRESS()
YADL_Dataset_Address.initialize()
YADL_Dataset_Address.freeze()
