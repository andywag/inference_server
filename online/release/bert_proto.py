from model_proto import *
import sys
sys.path.append("../models/bert")
sys.path.append("../models/bert/run")

import dataclasses

import numpy as np
import attr
from dataclasses import dataclass
from typing import List
import numpy as np



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
    
