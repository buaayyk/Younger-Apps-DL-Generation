#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-05-12 13:20:04
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import tqdm
import torch
import pathlib
import multiprocessing

from typing import Any, Callable, Literal

from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from torch_geometric.utils import is_sparse

from younger.commons.io import load_json, load_pickle
from younger.commons.utils import split_sequence
from younger.commons.logging import logger

from younger_logics_ir.modules import LogicX

from younger_apps_dl.datasets import register_dataset


class DAGData(Data):
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'level': # Directed Acyclic Graph Generation Level
            return 0

        if is_sparse(value) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'level': # Directed Acyclic Graph Generation Level
            return 0

        if 'batch' in key and isinstance(value, torch.Tensor):
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0


torch.serialization.add_safe_globals([DAGData, DataEdgeAttr, DataTensorAttr, GlobalStorage])


@register_dataset('dag')
class DAGDataset(Dataset):

    @property
    def raw_path(self):
        return self.raw_paths[0]

    @property
    def processed_path(self):
        return self.processed_paths[0]

    @property
    def raw_file_names(self):
        return [f'{self.raw_filename}']
        # return [f'{hash}' for hash in self.hashs]

    @property
    def processed_file_names(self):
        return [f'{self.processed_filename}']
        # return [f'{hash}' for hash in self.hashs]

    @property
    def raw_dir(self) -> str:
        return self.raw_dirpath

    @property
    def processed_dir(self) -> str:
        return self.processed_dirpath

    def len(self) -> int:
        return len(self.hashs)

    def get(self, index: int) -> DAGData:
        dag_data = self.all_dag_data[index]
        return dag_data

    def __init__(
        self,

        meta_filepath: str,
        raw_dirpath: str,
        raw_filename: str,
        processed_dirpath: str,
        processed_filename: str,
        name: str = 'YLDL-G2N',
        split: Literal['train', 'valid', 'test'] = 'train',
        worker_number: int = 4,

        root: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,
    ):
        # The class will not use `root`.
        # The `self.download()` will not be implemented, the function is delivered to preprocessors.

        self.meta_filepath = meta_filepath
        self.raw_dirpath = raw_dirpath
        self.raw_filename = raw_filename
        self.processed_dirpath = processed_dirpath
        self.processed_filename = processed_filename

        self.name = name
        self.split = split
        self.worker_number = worker_number

        self.meta = self.__class__.load_meta(self.meta_filepath)
        self.dicts = self.__class__.load_dicts(self.meta)
        self.hashs = self.__class__.load_hashs(self.meta)

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        logger.info('Loading Processed File')
        self.all_dag_data: list[DAGData] = torch.load(self.processed_path)

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        meta: dict[str, Any] = load_json(meta_filepath)
        return meta

    @classmethod
    def load_dicts(cls, meta: dict[str, Any]) -> dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]]:
        assert 'node_types' in meta
        assert isinstance(meta['node_types'], list)

        dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]] = dict()
        dicts['i2t'] = dict()
        dicts['t2i'] = dict()

        node_types = ['__UNK__'] + ['__MASK__'] + meta['node_types']
        for i, t in enumerate(node_types):
            dicts['i2t'][i] = t
            dicts['t2i'][t] = i
        return dicts

    @classmethod
    def load_hashs(cls, meta: dict[str, Any]) -> list[str]:
        assert 'item_names' in meta
        assert isinstance(meta['item_names'], list)
        hashs = sorted(meta['item_names'])
        return hashs


    def _process_chunk_(self, parameter: tuple[list[str], int]) -> list[DAGData]:
        sdags_chunk, worker_index = parameter
        dag_datas_chunk = list()
        with tqdm.tqdm(total=len(sdags_chunk), desc=f"Processing: Worker PID - {os.getpid()}", position=worker_index) as progress_bar:
            for sdag in sdags_chunk:
                dag = LogicX.loads_dag(sdag)
                dag_data = self.__class__.process_dag_data(dag, self.dicts)
                dag_datas_chunk.append(dag_data)
                progress_bar.update(1)
        return dag_datas_chunk

    def process(self):
        hash2sdag: dict[str, str] = load_pickle(self.raw_path)
        sdags = [hash2sdag[hash] for hash in self.hashs]
        chunk_count = self.worker_number
        sdags_chunks: list[list[str]] = split_sequence(sdags, chunk_count)
        worker_indices: list[int] = list(range(self.worker_number))
        dag_datas: list[DAGData] = list()
        with multiprocessing.Pool(self.worker_number) as pool:
            for dag_datas_chunk in pool.imap(self._process_chunk_, zip(sdags_chunks, worker_indices)):
                dag_datas.extend(dag_datas_chunk)

        torch.save(dag_datas, self.processed_path)

    @classmethod
    def process_dag_data(
        cls,
        dag, 
        dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]],
    ) -> DAGData:
        nxids = sorted(list(dag.nodes))
        pgids = list(range(dag.number_of_nodes()))
        nxid2pgid = dict(zip(nxids, pgids))
        # >>> print(list(sorted(logicx.dag.nodes)))
        # [B, A, C, D]
        # >>> print(nxid2pgid)
        # {A: 0, B: 1, C: 2, D: 3}

        x = cls.process_dag_x(dag, dicts, nxid2pgid)
        edge_index = cls.process_dag_edge_index(dag, nxid2pgid)
        if dag.graph['level']:
            level = cls.process_dag_level(dag, nxid2pgid)
            dag_data = DAGData(x=x, edge_index=edge_index, level=level)
        else:
            dag_data = DAGData(x=x, edge_index=edge_index)
        return dag_data

    @classmethod
    def process_dag_x(cls, dag, dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]], nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [#Node, 1]

        # ID in DAG
        node_indices_in_dag: list[str] = sorted(list(dag.nodes), key=lambda x: nxid2pgid[x])

        # ID in Dict
        node_indices_in_dict = list()
        for node_index_in_dag in node_indices_in_dag:
            node_uuid = dag.nodes[node_index_in_dag]['node_uuid']
            if node_uuid in dicts['t2i']:
                node_index_in_dict = [dicts['t2i'][node_uuid]]
            else:
                node_index_in_dict = [dicts['t2i']['__UNK__']]
            node_indices_in_dict.append(node_index_in_dict)
        x = torch.tensor(node_indices_in_dict, dtype=torch.long)

        return x

    @classmethod
    def process_dag_edge_index(cls, dag, nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [2, #Edge]
        edge_index = torch.empty((2, dag.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(list(dag.edges)):
            edge_index[0, index] = nxid2pgid[src]
            edge_index[1, index] = nxid2pgid[dst]
        return edge_index

    @classmethod
    def process_dag_level(cls, dag, nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [#Node, 1]

        level = list()
        node_indices_in_dag: list[str] = sorted(list(dag.nodes), key=lambda x: nxid2pgid[x])
        for index, node_index_in_dag in enumerate(node_indices_in_dag):
            level.append([dag.nodes[node_index_in_dag]['level']])
        level = torch.tensor(level, dtype=torch.long)
        return level
