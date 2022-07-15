
from pydantic.dataclasses import dataclass
from typing import List

@dataclass
class Squad:
    question:str
    answer:str

@dataclass
class SquadArray:
    text:List[List[str]]

@dataclass
class SquadResult:
    text:str
    logits_sum:float
    start:int
    end:int

@dataclass
class SquadResponse:
    results:List[SquadResult]
    server_time:float

@dataclass
class SquadArrayResponse:
    results:List[SquadResponse]
    server_time:float

@dataclass
class Ner:
    text:str

@dataclass
class NerArray:
    text:List[str]

@dataclass
class NerResult:
    start:int
    end:int
    index:int
    logit:float

@dataclass
class NerResponse:
    results:List[NerResult]
    server_time:float

@dataclass
class NerArrayResponse:
    results:List[NerResponse]
    server_time:float

@dataclass
class GPT2:
    text:str

@dataclass
class GPT2Response:
    results:str
    server_time:float

@dataclass
class Bart:
    text:str

@dataclass
class BartResponse:
    results:str
    server_time:float

@dataclass
class Dalli:
    text:str
    seed:int

@dataclass
class DalliResponse:
    results:List[List[List[int]]]
    server_time:float

@dataclass
class DalliResponseB64:
    results:str
    server_time:float