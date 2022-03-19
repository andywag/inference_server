from celery import Celery
import dataclasses
import dacite
from celery.utils.log import get_task_logger
from celery import states

import fine_tuning.celery_worker as fc
import offline.celery_worker as oc


app = Celery('infer', backend='rpc://', broker='pyamqp://192.168.3.114')
logger = get_task_logger(__name__)


@app.task(bind=True)
def run_fine(self, model_description_dict:dict, result_id:str):
    fc.run_fine(model_description_dict, result_id)    


@app.task(bind=True)
def run_infer(self, model_description_dict:dict, result_id:str):
    oc.run_infer(model_description_dict, result_id)
