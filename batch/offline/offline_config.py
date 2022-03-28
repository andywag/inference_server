from dataclasses import dataclass, field

@dataclass 
class Ipu:
    batches_per_step:int=64
    embedding_serialization_factor:int=1
    enable_half_partials:bool=True

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
    endpoint:str = ''

    ipu:Ipu = Ipu()
    detail:Detail=Detail()
    classifier:Classifier=Classifier()

    
    

