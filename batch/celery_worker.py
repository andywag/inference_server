from celery import Celery
import dataclasses
import dacite
from celery.utils.log import get_task_logger
from celery import states

from mongo_common import MongoInterface, get_mongo_interface

from offline.offline_config import InferDescription

import socket
from bson.objectid import ObjectId


app = Celery('infer', backend='rpc://', broker='pyamqp://192.168.3.114')


import logging
logger = logging.getLogger(__name__)

@app.task(bind=True)
def run_infer(self, model_description_dict:dict, result_id:str, train:bool=False):
    from offline.infer import main

    mongo = get_mongo_interface(ObjectId(result_id), train)
    self.update_state(state=states.STARTED)
    mongo.update_host(socket.gethostname())

    # TODO : Make Generic Support for Model
    model_description = dacite.from_dict(data_class=InferDescription, data=model_description_dict)
    result = main(model_description, train, mongo, self, logger)
    
    return result

