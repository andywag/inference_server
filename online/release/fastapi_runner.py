import sys
sys.path.append("../public_api")
sys.path.append("../model_proto")

from typing import Optional
from fastapi import FastAPI
from starlette.middleware.cors  import CORSMiddleware
import uvicorn

from api_classes import *
from rabbit_run_queue import RabbitRunQueue
from release_proto import models_map

app = FastAPI()
@app.get("/")
def home():
    return {"message":"Health Check Passed!"}


api_dict = {k:v.get_fast_apis()[0] for k,v in models_map.items()}



#ner_run_queue = RabbitRunQueue('ner')
@app.post("/squad_rabbit")
def run_ner(model_input:Squad) -> SquadResponse:
    return api_dict['squad'].run_rabbit(model_input)

@app.post("/ner_rabbit")
def run_ner(model_input:Ner) -> NerResponse:
    return api_dict['ner'].run_rabbit(model_input)

@app.post("/gpt2_rabbit")
def run_gpt2(model_input:GPT2) -> GPT2Response:
    return api_dict['gpt2'].run_rabbit(model_input)

@app.post("/bart_rabbit")
def run_bart(model_input:Bart) -> BartResponse:
    return api_dict['bart'].run_rabbit(model_input)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")