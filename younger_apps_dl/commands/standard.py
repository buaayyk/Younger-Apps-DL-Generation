#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 13:00:42
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-03 09:56:59
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_apps_dl.commands import equip_logger


@click.group(name='standard')
def standard():
    pass


@standard.group(name='train')
@click.option('--task-name', type=str)
@click.option('--config-filepath', type=str)
@click.option('--checkpoint-dirpath', type=str)
@click.option('--checkpoint-name', type=str, default='checkpoint')
@click.option('--keep-number', type=int, default=50)
@click.option('--train-batch-size', type=int, default=32)
@click.option('--valid-batch-size', type=int, default=32)
@click.option('--shuffle', action='store_true')
@click.option('--checkpoint-filepath', type=str, default=None)
@click.option('--reset-optimizer', action='store_true')
@click.option('--reset-period', action='store_true')
@click.option('--life-cycle', type=int, default=100)
@click.option('--report-period', type=int, default=100)
@click.option('--update-period', type=int, default=1)
@click.option('--train-period', type=int, default=1000)
@click.option('--valid-period', type=int, default=1000)
@click.option('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
@click.option('--world-size', type=int, default=1)
@click.option('--master-addr', type=str, default='localhost')
@click.option('--master-port', type=str, default='16161')
@click.option('--master-rank', type=int, default=0)
@click.option('--seed', type=int, default=1234)
@click.option('--make-deterministic', action='store_true')
def standard_train():
    pass
