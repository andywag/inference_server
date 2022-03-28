

from datasets import load_dataset, load_from_disk
from adlfs import AzureBlobFileSystem

imdb = load_dataset('imdb')

fs = AzureBlobFileSystem(connection_string="DefaultEndpointsProtocol=https;AccountName=andynlpstore;AccountKey=hkMiWLiqIpONH0NnyhmYAO9SmdVJZb1CazjCB6mnk/72ee5KdyKnq/ByS5s6/ZPUPbP2HImIveIvwxYSP88Reg==;EndpointSuffix=core.windows.net")
result = fs.ls("graphcore")
print(result)

base = load_from_disk('graphcore/masked_test',fs=fs)



