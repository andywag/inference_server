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
from bert_config_new import BertDescription

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
    classifier:str
    num_labels:int

    def create_model_description(self, model_description):
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

@app.post("/tune")
def run_tune(model_input:ModelConfig) -> ModelResponse:

    mongo = MongoInterface()

    if model_input.model_type == 'BERT':
        mongo.update_status(result_id,"Submit")
        model_description = BertDescription()
        model_description = model_input.create_model_description(bert_description)
        model_description.model_specific.tuning_type = model_input.classifier
        model_description.model_specific.num_labels = model_input.num_labels

        #model_description = model_input.create_model_description()
    else:
        mongo.update_status(result_id,"ModelNotSupported")
        print("Model Not Supported")
    
    result = ModelResult("", "", model_description, list(), list())
    result_dict = dataclasses.asdict(result)
    result_id = mongo.create_result(result_dict)
    
    print(model_input)
    result = run_dict.delay(model_description, str(result_id))

    return ModelResponse(str(result_id))
    # Create the Model
    

    #model_description_dict = dataclasses.asdict(model_description)
    #bert_specific = BertSpecific("bert")
    #bert_specific.tuning_type = model_input.classifier
    #bert_specific.num_labels = model_input.num_labels
    #bert_specific_dict = dataclasses.asdict(bert_specific)
    #print("Here", bert_specific.num_labels)
    # Create the Model Result
    #result = ModelResult("", "", model_description, list(), list())
    #result_dict = dataclasses.asdict(result)
    #mongo = MongoInterface()
    #result_id = mongo.create_result(result_dict)
    # Update the Mongo Status
    #mongo.update_status(result_id,"Submitting")
    #result = run_dict.delay(model_description_dict, bert_specific_dict, str(result_id))


    #return ModelResponse("Started")




uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")