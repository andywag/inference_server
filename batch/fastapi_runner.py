from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dataclasses import dataclass
import dataclasses
import pymongo

import offline_runner
from infer_config import InferConfig
from mongo_common import get_infer_results, get_final_results, get_fine_results

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
    return get_fine_results()
    #return fine_runner.get_results()

@app.post("/tune")
def run_tune(model_input:InferConfig):
    return offline_runner.run(model_input, True)
   

@app.get("/infer_results")
def get_results():
    return get_infer_results()

@app.post("/infer")
def run_infer(model_input:InferConfig) :
    return offline_runner.run(model_input)
    
@app.get("/infer_final_result")
def get_fin_results(id):
    return get_final_results(id)



uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")