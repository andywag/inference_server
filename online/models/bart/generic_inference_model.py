
import poptorch
import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.optimized_gpt2_attn import OptimizedGPT2Attention_test, OptimizedGPT2Attention_nobuffer

from dataclasses import dataclass, field
from typing import List
import os

def get_cache_dir():
    build_dir =  os.getenv('INFERENCE_BUILD')
    if build_dir is None:
        println("INFERENCE_BUILD not set")
        build_dir = "."
    cache_dir = f"{build_dir}/cache"
    

@dataclass
class Config:
    custom_ops:bool = False
    ipus_per_replica:int = 1
    replication_factor:int = 1
    batches_per_step:int = 1
    executable_cache_dir:str = get_cache_dir()
    enable_half_partials:bool = True
    use_popdist:bool = False
    matmul_proportion:List = field(default_factory = lambda: [0.2])
    seed:int = 10
    gradient_accumulation:bool = False
    profile:bool=False
    



