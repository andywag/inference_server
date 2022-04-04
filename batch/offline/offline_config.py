from dataclasses import dataclass, field
from typing import List

@dataclass 
class Ipu:
    batches_per_step:int=64
    embedding_serialization_factor:int=1
    enable_half_partials:bool=True
    training:bool=False
    auto_loss_scaling:bool=False
    optimizer_state_offchip:bool=False
    replicated_tensor_sharding:bool=True
    enable_half_partials:bool=True
    use_popdist:bool=False
    recompute_checkpoint_every_layer:bool = True
    ipus_per_replica:int=4
    layers_per_ipu:List[float]=field(default_factory=lambda: [0, 4, 4, 4])
    matmul_proportion:List[float]=field(default_factory=lambda: [0.25,0.25,0.25,0.25])


@dataclass 
class Detail:
    sequence_length:int=384
    batch_size:int=16

@dataclass
class Classifier:
    classifier_type:str='Sequence'
    num_labels:int=2

@dataclass
class InferDescription:
    name:str=''
    dataset:str='imdb'
    tokenizer:str='bert-base-uncased'
    model:str='bert-base-uncased'
    checkpoint:str='textattack/bert-base-uncased-imdb'
    cloud:str=''
    endpoint:str = ''
    result_folder:str=''

    ipu:Ipu = Ipu()
    detail:Detail=Detail()
    classifier:Classifier=Classifier()

    
    

