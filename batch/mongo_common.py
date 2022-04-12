from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar, List
import pymongo
from dacite import from_dict
import time
from bson.int64 import Int64
import dataclasses
from bson.objectid import ObjectId

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
    accuracy:float=0.0
    qps:float=0.0
    results:List[Result]=field(default_factory=lambda: [])
    status:List[StatusLog]=field(default_factory=lambda: [StatusLog("Submitted",Int64(time.time()))])

client = pymongo.MongoClient("mongodb://192.168.3.114:27017/")
db = client.run_database
infer = db.infer
fine = db.fine
infer_results = db.infer_result


class MongoInterface:

    def __init__(self, collection, result_collection, result_id=None):
        self.client = pymongo.MongoClient("mongodb://192.168.3.114:27017/")
        self.db = self.client.run_database
        self.collection = self.db[collection]
        self.result_collection = self.db[result_collection]
        self.result_id = ObjectId(result_id)

    def update_accuracy(self, accuracy=0.0, qps=0.0):
        self.collection.update_one({"_id": self.result_id},
            {"$set":  {"accuracy":  accuracy, "qps":qps} } 
        )

    def update_loss(self, loss=0.0, qps=0.0):
        self.collection.update_one({"_id": self.result_id},
            {"$set":  {"loss":  loss, "qps":qps} } 
        )
   

    def update_id(self, uuid):
        self.collection.update_one({"_id": self.result_id},
            {"$set":  {"uuid":  uuid} } 
        )

    def update_id(self,  uuid):
        self.collection.update_one({"_id": self.result_id},
            {"$set":  {"uuid":  uuid} } 
        )

    def update_host(self, hostname):
        self.collection.update_one({"_id": self.result_id},
            {"$set":  {"hostname":  hostname} } 
        )


    def create_result(self, result_dict):
        result_id = self.collection.insert_one(result_dict).inserted_id
        return result_id

    def update_status(self, value, message=None):
        self.collection.update_one({"_id": self.result_id},
            {"$push":  {"status":  {"status":value, "time":Int64(time.time()),"detail":message}} } 
        )
        

    def update_result(self, value):
        self.collection.update_one({"_id": self.result_id},
            {"$push":  {"results":  value } } 
        )

    def put_result(self, results_dict):
        test_result = {'result_id':self.result_id, 'results':results_dict}
        self.result_collection.insert_one(test_result)


client = pymongo.MongoClient("mongodb://192.168.3.114:27017/")
db = client.run_database
infer = db.infer
fine = db.fine
infer_results = db.infer_result

def get_infer_result(self, result_id):
    return infer.find_one({'_id':result_id})

def get_infer_results():
    cursor = infer.find({})
    documents = []
    for document in cursor:
        document['sid'] = str(document['_id'])
        del document['_id']
        documents.append(document)
    return documents

def get_fine_result(self, result_id):
    return fine.find_one({'_id':result_id})

def get_fine_results():
    cursor = fine.find({})
    documents = []
    for document in cursor:
        document['sid'] = str(document['_id'])
        del document['_id']
        documents.append(document)
    return documents
    

def get_table_name(train:bool=False):
    mongo_table = "infer"
    if train:
        mongo_table="fine"
    return mongo_table

def create_mongo_interface(model_description:T, train:bool=False):
    table = get_table_name(train)
    mongo = MongoInterface(table, f"{table}_result")
    result = ModelResult(model_description)
    result_dict = dataclasses.asdict(result)
    result_id = mongo.create_result(result_dict)
    mongo.update_status(result_id,"Submit")
    return mongo, result_id

def get_final_results(id):
    document = infer_results.find_one({'result_id':ObjectId(id)})
    #print("Document", document)
    if document is not None:
        del document['_id']
        del document['result_id']
    return document

