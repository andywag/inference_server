from model_proto import *
import sys
sys.path.append("./bert")
sys.path.append("./bert/run")

from bert_interface_wrapper import BertInterfaceWrapper

import numpy as np
import attr
from dataclasses import dataclass
from model_proto import SignalProto


input_ids = SignalProto('input_ids',np.zeros(shape=384, dtype=np.uint32))
segment_ids = SignalProto('segment_ids',np.zeros(shape=1, dtype=np.uint32))
query_ids = SignalProto('query_ids',np.zeros(shape=1, dtype=np.uint64))

query_ids_result = SignalProto('query_ids_result',np.zeros(shape=1, dtype=np.uint64))
