
from transformers import AutoTokenizer, AutoConfig
from fine_tune_config import ModelDescription, ModelResult
from ipu_options import get_options
from datasets import load_dataset
from optimization import get_optimizer

import poptorch
from celery import states
import pymongo
import dataclasses
from bson.objectid import ObjectId

result = ModelResult("test", ModelDescription(), list())
result_dict = dataclasses.asdict(result)

client = pymongo.MongoClient("mongodb://192.168.3.114:27017/")
db = client.run_database
collection = db.fine_tuning
    
result_id = collection.insert_one(result_dict).inserted_id
print(f"Result {result_id} {type(result_id)}")

x = collection.find_one({"_id": result_id})
print("Found", x)

#x = collection.find_one({"uuid": "test"})
#print("Found", x)

ret = collection.update_one({"_id": result_id},
            {"$push":  {"results":  0.0 } } 
        )
print("Find", ret, ret.matched_count, ret.modified_count)