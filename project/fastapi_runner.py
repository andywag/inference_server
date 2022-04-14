import sys
sys.path.append("../public_api")

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from release_proto import project_proto
import uvicorn

from general_fastapi import get_apis, get_apis_dict 
from api_classes import *
from rabbit_run_queue import RabbitRunQueue

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

apis = get_apis(project_proto)
api_dict = get_apis_dict(project_proto)

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


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

@app.post("/bart")
def run_bart(model_input:Bart) -> BartResponse:
    return apis[0].run(model_input)

@app.post("/ner")
def run_ner(model_input:Ner) -> NerResponse:
    return apis[1].run(model_input)

@app.post("/gpt")
def run_gpt2(model_input:GPT2) -> GPT2Response:
    return apis[2].run(model_input)


uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")