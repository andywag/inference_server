
from adlfs import AzureBlobFileSystem
from datasets import load_from_disk

class CloudFileContainer:
    
    def __init__(self, cloud_type:str, endpoint:str):
        cloud_file_system = None
        if cloud_type is not None and cloud_type != '' and cloud_type != 'None':
            if cloud_type == 'AzureBlob':
                cloud_file_system = AzureBlobFileSystem(connection_string=endpoint)
            else:
                print("Cloud File System Not Supported")
            self.cloud_file_system = cloud_file_system
        else:
            self.cloud_file_system = None

    def load_dataset(self, data_tag):

        try:
            if self.cloud_file_system is not None:
                data_tag = data_tag.split(",")
                dataset = load_from_disk(data_tag[0], fs = self.cloud_file_system)
        except Exception as e:
            print("Error", e)

        if dataset is None:
            data_internal = data_tag[0].split(":")
            if len(data_internal) == 1:
                dataset = load_dataset(data_internal[0])
            else:
                dataset = load_dataset(data_internal[0], data_internal[1])
        
        for tag in data_tag[1:-1]:
            dataset = dataset[tag]
    
        if data_tag[-1] != 'text':
            dataset = dataset.map(lambda x:{'text':data_tag[-1]})
        
        return dataset

    def output_file(self, result_file, results):
        if self.cloud_file_system is not None:
            with self.cloud_file_system.open(result_file, 'w') as fp:
                json.dump(results, fp)
