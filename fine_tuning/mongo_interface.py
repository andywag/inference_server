    
import pymongo
from dacite import from_dict
from fine_tune_config import ModelResult, StatusLog
import time
from bson.int64 import Int64

class MongoInterface:

    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://192.168.3.114:27017/")
        self.db = self.client.run_database
        self.collection = self.db.fine_tuning

    def create_result(self, result_dict):
        result_id = self.collection.insert_one(result_dict).inserted_id
        return result_id

    def update_status(self, result_id, value):
        self.collection.update_one({"_id": result_id},
            {"$push":  {"status":  {"status":value, "time":Int64(time.time()) }} } 
        )
        

    def update_result(self, result_id, value):
        self.collection.update_one({"_id": result_id},
            {"$push":  {"results":  value } } 
        )

    def get_all_results(self):
        cursor = self.collection.find({}).sort("_id",-1) 
        documents = []
        for document in cursor:
            #document['_id'] = str(document['_id'])
            del document['_id']
            document = from_dict(data_class=ModelResult, data=document)
            documents.append(document)
        print("Returning Documnents", len(documents))
        return documents
        #print("Find", ret, ret.matched_count, ret.modified_count)