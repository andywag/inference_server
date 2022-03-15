from pydantic.dataclasses import dataclass
import requests
from typing import List
import dataclasses

from api_classes import *


base_url = "http://192.168.3.114:8100"

squad_url = f"{base_url}/squad"
squad_array_url = f"{base_url}/squad_array"
ner_url = f"{base_url}/ner"
ner_array_url = f"{base_url}/ner_array"



def post_squad(text:Squad) -> SquadResponse:
    squad_request = dataclasses.asdict(text)
    result =  requests.post(squad_url, json = squad_request)
    #print("A", result.text)
    return result

def post_squad_array(text:SquadArray) -> SquadArrayResponse:
    squad_request = dataclasses.asdict(text)
    result =  requests.post(squad_array_url, json = squad_request)
    return result

def post_ner(text:Ner)->NerResponse:
    ner_request = dataclasses.asdict(text)
    result =  requests.post(ner_url, json = ner_request)
    return result

def post_ner_array(text:NerArray) -> NerArrayResponse:
    squad_request = dataclasses.asdict(text)
    result =  requests.post(ner_array_url, json = squad_request)
    return result

def post(query):
    if isinstance(query, Squad):
        post_squad(query)
    elif isinstance(query, SquadArray):
        post_squad_array(query)
    elif isinstance(query, Ner):
        post_ner(query)
    elif isinstance(query, NerArray):
        post_ner_array(query)

