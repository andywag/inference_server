    
import pymongo
from dacite import from_dict
from .fine_tune_config import ModelResult, StatusLog
import time
from bson.int64 import Int64

class MongoInterface:

    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://192.168.3.114:27017/")
        self.db = self.client.run_database
        self.collection = self.db.fine_tuning

    def update_accuracy(self, result_id, accuracy=0.0, qps=0.0):
        self.collection.update_one({"_id":result_id},
            {"$set":  {"loss":  accuracy, "qps":qps} } 
        )

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
