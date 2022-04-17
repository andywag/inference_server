
import poptorch
import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.optimized_gpt2_attn import OptimizedGPT2Attention_test, OptimizedGPT2Attention_nobuffer

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    custom_ops:bool = False
    ipus_per_replica:int = 1
    replication_factor:int = 1
    batches_per_step:int = 1
    executable_cache_dir:str = "./cache_exe"
    enable_half_partials:bool = True
    use_popdist:bool = False
    matmul_proportion:List = field(default_factory = lambda: [0.2])
    seed:int = 10
    gradient_accumulation:bool = False
    profile:bool=False
    

class TorchModelWrapper(nn.Module):
    def __init__(self, model, convert_input, convert_output):
        super().__init__()
        self.model = model
        self.convert_output = convert_output

    def forward(self, input_data):
        #inputs = {
        #    "input_ids": input_ids,
        #}
        #output = super().forward(**inputs)
        output = self.model(**input_data)
        return self.convert_output(output) 
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        return model


class ModelWrapper:
    def __init__(self, base_model):
        self.base_model = base_model
        self.model = self.create_ipu(base_model)

    def create_ipu(self, base_model):
        opts = poptorch.Options()
        opts.setAvailableMemoryProportion({'IPU0': 0.2})
        opts.autoRoundNumIPUs(True)
        base_model.half()
        base_model.eval()
        model = poptorch.inferenceModel(base_model, opts)
        return model

    def run_data(self, input):
        return self.model(input)
