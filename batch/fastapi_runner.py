from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dataclasses import dataclass
import dataclasses
from fine_tuning.fine_tune_config import ModelDescription, ModelResult
from fine_tuning.runner import ModelConfig, ModelResponse
import pymongo

import fine_tuning 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class InferDescription:
    name:str
    model_type:str
    checkpoint:str
    tokenizer:str
    dataset:str
    classifier:str
    num_labels:int

    def create_model_description(self):
        
        model_description.name = self.name
        model_description.tokenizer = self.tokenizer
        model_description.checkpoint = self.checkpoint
        model_description.dataset = self.dataset
        bert_description.model_specific.tuning_type = model_input.classifier
        bert_description.model_specific.num_labels = model_input.num_labels
        
        return model_description



@app.get("/results")
def get_results():
    return fine_tuning.runner.get_results()

@app.post("/tune")
def run_tune(model_input:ModelConfig) -> ModelResponse:
    return fine_tuning.runner.run(model_input)
   
    # Create the Model
    




uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")