from model_proto import *
import sys
sys.path.append("./bert")
sys.path.append("./bert/run")

import dataclasses
from bert_interface_wrapper import BertInterfaceWrapper

import numpy as np
import attr
from dataclasses import dataclass
from model_proto import SignalProto
from typing import List
import numpy as np


input_ids = SignalProto('input_ids',np.zeros(shape=384, dtype=np.uint32))
segment_ids = SignalProto('segment_ids',np.zeros(shape=1, dtype=np.uint32))
query_ids = SignalProto('query_ids',np.zeros(shape=1, dtype=np.uint64))

query_ids_result = SignalProto('query_ids_result',np.zeros(shape=1, dtype=np.uint64))

@dataclass
class BertInput:
    input_ids:List[List[int]]
    segment_id:List[List[int]]
    query_id:List[int]

    def items(self):
        return [self.input_ids, self.segment_id, self.query_id]

@dataclass
class NerOutput:
    logits:List[List[float]]

    @staticmethod
    def create(data:np.ndarray):
        return NerOutput(data.tolist())
    
