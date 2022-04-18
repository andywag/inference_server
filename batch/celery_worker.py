from celery import Celery
import dataclasses
import dacite
from celery.utils.log import get_task_logger
from celery import states

from mongo_common import MongoInterface, get_mongo_interface

from offline.offline_config import InferDescription
from offline.infer import main
#from fine_tuning.fine_tune import main as fine_main

import socket
from bson.objectid import ObjectId
#from fine_tuning.bert_model.bert_config_new import BertDescription

#import poptorch

app = Celery('infer', backend='rpc://', broker='pyamqp://192.168.3.114')
#logger = get_task_logger(__name__)


import logging
logger = logging.getLogger(__name__)
#logger = None

@app.task(bind=True)
def run_infer(self, model_description_dict:dict, result_id:str, train:bool=False):
    mongo = get_mongo_interface(ObjectId(result_id), train)
    self.update_state(state=states.STARTED)
    mongo.update_host(socket.gethostname())

    # TODO : Make Generic Support for Model
    model_description = dacite.from_dict(data_class=InferDescription, data=model_description_dict)
    result = main(model_description, train, mongo, self, logger)
    
    return result

