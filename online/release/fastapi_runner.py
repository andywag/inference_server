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


# 
# FIXME : Models Need to be directly added at the file top level
#
#for api in apis:
#    print(f"Creating API {api.path} {api.input_type} {api.output_type}")
#    @app.post(f"/{api.path}", name=api.path)
#    @rename(api.path)
#    def runner(input_data:api.input_type) -> api.output_type:
#        print("Running Fast API", api.path)
#        return api.run(input_data)

#@app.post("/squad")
#def run_squad(model_input:Squad) -> SquadResponse:
#    return apis[0].run(model_input)

#@app.post("/bart")
#def run_bart(model_input:Bart) -> BartResponse:
#    return apis[0].run(model_input)

#@app.post("/ner")
#def run_ner(model_input:Ner) -> NerResponse:
#    return apis[1].run(model_input)

#@app.post("/gpt")
#def run_gpt2(model_input:GPT2) -> GPT2Response:
#    return apis[2].run(model_input)


uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")