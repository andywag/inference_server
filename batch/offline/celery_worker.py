from celery import Celery
import dataclasses
import dacite
from celery.utils.log import get_task_logger
from celery import states
from .offline_config import InferDescription
from .infer import main

app = Celery('infer', backend='rpc://', broker='pyamqp://192.168.3.114')
logger = get_task_logger(__name__)




#@app.task(bind=True)
def run_infer(self, model_description_dict:dict, result_id:str):
    self.update_state(state=states.STARTED)

    # TODO : Make Generic Support for Model
    model_description = dacite.from_dict(data_class=InferDescription, data=model_description_dict)
    logger.info(f"Model{model_description}")

    result =  main(model_description, result_id, self, logger)
    return result
