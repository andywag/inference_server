from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Optional
import dataclasses
from datetime import datetime

@unique            
class OptimizerEnum(str, Enum):
    ADAM = "Adam"
    LAMB = "LAMB"
    LAMB_NO_BIAS = "LAMBNoBiasCorrection"

@dataclass
class OptimizerDescription:
    optimizer_type:str = "Adam"
    lr_warmup:float=0.28
    learning_rate:float=0.00005
    loss_scaling:float=16.0
    weight_decay:float=0.01
    enable_half_first_order_momentum:bool=False

@dataclass
class ExecutionDescription:
    batch_size:int=4
    batches_per_step:int=1
    replication_factor:int=1
    gradient_accumulation:int=32
    random_seed:int=42
    training_steps:Optional[int]=None
    epochs:Optional[int]=None

@dataclass
class IpuOptions:
    replicated_tensor_sharding:bool=False
    auto_loss_scaling:bool=False
    optimizer_state_offchip:bool=False
    replicated_tensor_sharding:bool=True
    enable_half_partials:bool=True
    use_popdist:bool=False
    executable_cache_dir:str='./cache_dir'
    profile:bool=False
    profile_dir:str='./profile_dir'
    recompute_checkpoint_every_layer:bool = True

@dataclass
class IpuLayout:
    ipus_per_replica:int=4
    layers_per_ipu:List[float]=field(default_factory=lambda: [0, 4, 4, 4])
    matmul_proportion:List[float]=field(default_factory=lambda: [0.25,0.25,0.25,0.25])

@dataclass
class ModelDataset:
    name:str="imdb"
    train:Optional[str]="train"
    text:str="text"
    token_format:List[str]=field(default_factory=lambda:['input_ids', 'token_type_ids', 'attention_mask', 'label'])

@dataclass
class ModelSpecific:
    name:str

    def get_model(self, checkpoint):
        raise NotImplementedError("Need to Generate Model")

@dataclass
class ModelDescription:
    name:str="bert"
    tokenizer:str="bert-base-uncased"
    checkpoint:str="bert-base-uncased"
    ipu_options:IpuOptions=IpuOptions()
    optimizer_description:OptimizerDescription=OptimizerDescription()
    execution_description:ExecutionDescription=ExecutionDescription()
    ipu_layout:IpuLayout=IpuLayout()
    dataset:ModelDataset=ModelDataset()
    model_specific:ModelSpecific=ModelSpecific('test')


@dataclass
class Result:
    epoch:int
    time:float
    error:float
    accuracy:float

@dataclass
class StatusLog:
    status:str
    time:int

@dataclass
class ModelResult:
    uuid:str
    hostname:Optional[str]
    description:ModelDescription
    results:List[Result]
    status:List[StatusLog]
