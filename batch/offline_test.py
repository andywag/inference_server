
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("conll2003")
dataset = dataset["train"]
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


res = tokenizer.convert_tokens_to_ids(['graphing','Iyer'])
tos = tokenizer.convert_ids_to_tokens(res)

print("A", res, tos)

def create_tokens(data, mlen=384):
    def internal(x):
        new_result =[0]*mlen
        new_result[0]=101 
        result = tokenizer.convert_tokens_to_ids(x)
        new_len = min(mlen, len(result))
        new_result[1:new_len+1] = result[:new_len]
        if len(result) < mlen-2:
            new_result[len(result)+1]=102
        return new_result
    result = [internal(x) for x in data]
    return result

    
def create_labels(data, mlen):
    new_results = [0]*mlen
    new_results[1:len(data)+1] = data
    return new_results
   
dataset = dataset.map(lambda x: {'labels':create_labels(x['ner_tags'],384)})
print("Asdf", dataset[0]['labels'])

tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
    max_length=self.inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True,return_offsets_mapping=True,return_length=True),batched=True)


result = create_labels(dataset['tokens'])

print("A", dataset[0]['tokens'])
print("B", result[0])
print("C", dataset[0]['ner_tags'])
