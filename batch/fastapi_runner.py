from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dataclasses import dataclass
import dataclasses
from fine_tuning.fine_tune_config import ModelDescription, ModelResult
from fine_tuning.runner import ModelConfig, ModelResponse
import pymongo

from offline.runner import InferConfig

import fine_tuning 
import offline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





@app.get("/results")
def get_results():
    return fine_tuning.runner.get_results()

@app.post("/tune")
def run_tune(model_input:ModelConfig) -> ModelResponse:
    return fine_tuning.runner.run(model_input)
   
@app.post("/infer")
def run_tune(model_input:InferConfig) :
    return offline.runner.run(model_input)
    # Create the Model
    




uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")