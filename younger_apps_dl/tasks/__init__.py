#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-17 03:29:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-09 10:34:58
# Copyright (c) 2024 - 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from younger_apps_dl.tasks.base_task import YoungerTask
from younger_apps_dl.tasks.performance_prediction import PerformancePrediction
from younger_apps_dl.tasks.block_embedding import BlockEmbedding
from younger_apps_dl.tasks.node_prediction import NodePrediction
from younger_apps_dl.tasks.link_prediction import LinkPridiction
from younger_apps_dl.tasks.node_embedding import NodeEmbedding
from younger_apps_dl.tasks.ssl_prediction import SSLPrediction


task_builders: dict[str, YoungerTask] = dict(
    performance_prediction = PerformancePrediction,
    block_embedding = BlockEmbedding,
    node_prediciton = NodePrediction,
    link_prediction = LinkPridiction,
    node_embedding = NodeEmbedding,
    ssl_prediction= SSLPrediction,
)
