from celery import Celery
from fine_tune_config import ModelDescription
#from bert_model.bert_config import BertSpecific
from fine_tune import main
import dataclasses
import dacite
from celery.utils.log import get_task_logger
from celery import states
from bert_model.bert_config_new import BertDescription

app = Celery('fine_tuning', backend='rpc://', broker='pyamqp://192.168.3.114')
#app = Celery('fine_tuning', backend='redis://192.168.3.114', broker='pyamqp://192.168.3.114')
logger = get_task_logger(__name__)




@app.task(bind=True)
def run_dict(self, model_description_dict:dict, result_id:str):
    self.update_state(state=states.STARTED)

    # TODO : Make Generic Support for Model
    model_description = dacite.from_dict(data_class=BertDescription, data=model_description_dict)
    logger.info(f"Model{model_description}")

    #return "HHH"
    result =  main(model_description, result_id, self, logger)
    return result
