

from adlfs import AzureBlobFileSystem
from cloud_utils import CloudFileContainer


endpoint = "DefaultEndpointsProtocol=https;AccountName=andynlpstore;AccountKey=hkMiWLiqIpONH0NnyhmYAO9SmdVJZb1CazjCB6mnk/72ee5KdyKnq/ByS5s6/ZPUPbP2HImIveIvwxYSP88Reg==;EndpointSuffix=core.windows.net"
container = CloudFileContainer('AzureBlob', endpoint=endpoint)

container.store_directory('temp', 'graphcore')