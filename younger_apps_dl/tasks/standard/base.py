#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-13 22:30:53
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-14 13:03:37
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from younger_apps_dl.tasks.base import BaseTask


class StandardTask(BaseTask):

    checkpoint = dict()
    checkpoint['Epoch'] = epoch
    checkpoint['Step'] = step
    checkpoint['model_state'] = task.model.module.state_dict() if is_distribution else task.model.state_dict()
    checkpoint['optimizer_state'] = task.optimizer.state_dict()
    save_checkpoint(checkpoint, checkpoint_path=checkpoint_dirpath, checkpoint_name=checkpoint_name, keep_number=keep_number)