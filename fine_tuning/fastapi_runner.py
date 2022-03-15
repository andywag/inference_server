from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dataclasses import dataclass
from celery_worker import  run_dict
import dataclasses
from fine_tune_config import ModelDescription, ModelResult
from bert_model.bert_config import BertSpecific
import pymongo
from mongo_interface import MongoInterface

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class ModelConfig:
    name:str
    model_type:str
    checkpoint:str
    dataset:str
    tokenizer:str
    optimizer:str
    learning_rate:float
    epochs:int

    def create_model_description(self):
        model_description = ModelDescription()
        model_description.name = self.name
        model_description.tokenizer = self.tokenizer
        model_description.checkpoint = self.checkpoint
        model_description.execution_description.epochs = self.epochs
        model_description.execution_description.learning_rate = self.learning_rate
        return model_description

@dataclass
class ModelResponse:
    request_id:str

@dataclass
class ModelResults:
    results:List[ModelResult]

@app.get("/results")
def get_results() -> ModelResults:
    mongo = MongoInterface()
    results = mongo.get_all_results()
    return ModelResults(results)
    #return results

@app.post("/tune")
def run_bart(model_input:ModelConfig) -> ModelResponse:
    print(model_input)
    model_description = model_input.create_model_description()
    model_description_dict = dataclasses.asdict(model_description)
    bert_specific_dict = dataclasses.asdict(BertSpecific("bert"))

    result = run_dict.delay(model_description_dict, bert_specific_dict)
    

    return ModelResponse("Here I am ...")




uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")