
from fine_tune_config import ModelDescription
from bert_model.bert_config import BertSpecific
from fine_tune import main
from celery_worker import  run_dict
import dataclasses

model_description = ModelDescription()
model_description_dict = dataclasses.asdict(model_description)
bert_specific_dict = dataclasses.asdict(BertSpecific("bert"))
#print("BBBB", model_description_dict, BertSpecific())

#r = run.delay(model_description)
#print("A",r )

#print("A", model_description_dict)
#main(model_description)
result = run_dict.delay(model_description_dict, bert_specific_dict)

#result = hello.delay()
print(result)
print(result.get())