from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar, List
import pymongo
from dacite import from_dict
import time
from bson.int64 import Int64
import dataclasses

@dataclass
class Result:
    epoch:int
    time:float
    error:float
    accuracy:float

@dataclass
class StatusLog:
    status:str
    time:int
    detail:Optional[str]=None

T = TypeVar("T")

@dataclass
class ModelResult(Generic[T]):
    description:T
    uuid:str=""
    hostname:str=""
    results:List[Result]=field(default_factory=lambda: [])
    status:List[StatusLog]=field(default_factory=lambda: [])


class MongoInterface:

    def __init__(self, collection):
        self.client = pymongo.MongoClient("mongodb://192.168.3.114:27017/")
        self.db = self.client.run_database
        self.collection = self.db[collection]

    def update_id(self, result_id, uuid):
        self.collection.update_one({"_id": result_id},
            {"$set":  {"uuid":  uuid} } 
        )

    def update_host(self, result_id, hostname):
        self.collection.update_one({"_id": result_id},
            {"$set":  {"hostname":  hostname} } 
        )


    def create_result(self, result_dict):
        result_id = self.collection.insert_one(result_dict).inserted_id
        return result_id

    def update_status(self, result_id, value, message=None):
        self.collection.update_one({"_id": result_id},
            {"$push":  {"status":  {"status":value, "time":Int64(time.time()),"detail":message}} } 
        )
        

    def update_result(self, result_id, value):
        self.collection.update_one({"_id": result_id},
            {"$push":  {"results":  value } } 
        )

    def get_result(self, result_id):
        return self.collection.find_one({'_id':result_id})

    def get_all_results(self):
        cursor = self.collection.find({})
        documents = []
        for document in cursor:
            del document['_id']
            documents.append(document)
        return documents

def create_mongo_interface(model_description:T):
    mongo = MongoInterface("infer")
    result = ModelResult(model_description)
    result_dict = dataclasses.asdict(result)
    result_id = mongo.create_result(result_dict)
    mongo.update_status(result_id,"Submit")
    return mongo, result_id
