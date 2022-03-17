from celery import Celery
from fine_tune_config import ModelDescription
from bert_model.bert_config import BertSpecific
from fine_tune import main
import dataclasses
import dacite
from celery.utils.log import get_task_logger
from celery import states

app = Celery('fine_tuning', backend='rpc://', broker='pyamqp://192.168.3.114')
#app = Celery('fine_tuning', backend='redis://192.168.3.114', broker='pyamqp://192.168.3.114')
logger = get_task_logger(__name__)




@app.task(bind=True)
def run_dict(self, model_description_dict:dict, specific_description_dict:dict, result_id:str):
    #model_description_dict
    #logger.info(f"Running Model {model_description_dict} {specific_description_dict}")
    print(specific_description_dict)
    self.update_state(state=states.STARTED)
    #model_description_dict['model_specific'] = specific_description_dict
    #print("Model", model_description_dict, type(model_description_dict))
    #ipu_options = model_description_dict['ipu_options']
    #ipu_options = model_description_dict['ipu_layout']

    #print("BBB", ipu_options)
    model_description = dacite.from_dict(data_class=ModelDescription, data=model_description_dict)
    #logger.info("Base", dataclasses.asdict(model_description))
    specific_description = dacite.from_dict(data_class=BertSpecific, data=specific_description_dict)
    model_description.model_specific = specific_description
    print("Model", model_description)

    #return "HHH"
    result =  main(model_description, self, logger)
    return result
