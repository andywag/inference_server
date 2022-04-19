
from typing import Optional, List
from dataclasses import dataclass
import dataclasses
import pymongo

from offline.offline_config import InferDescription
from mongo_common import create_mongo_interface

from infer_config import InferConfig, ModelResponse


def run(model_input:InferConfig, train:bool=False) -> ModelResponse:
    from celery_worker import  run_infer

    model_description = model_input.create_model_description()
    mongo, result_id = create_mongo_interface(model_description,  train)

    model_description_dict = dataclasses.asdict(model_description)
    #if train:
    uuid = run_infer.delay(model_description_dict, str(result_id), train)
    #else:
    #    uuid = run_infer.delay(model_description_dict, str(result_id))

    # Attach the ID to the Database
    mongo.update_id(str(uuid))
    mongo.update_status("Submitted")
    print("Running Inference", result_id, train)
    return ModelResponse(str(result_id))

