from sprucepy import secrets
from azure.storage.blob import BlobServiceClient 

import yaml
import io

from . import constants
    
def get_secrets():
    return {
        k: secrets.get_secret_by_key(k, api_url=constants.SPRUCE_API_URL)
        for k in constants.SECRETS
    }
    
def container_conn( ):
    """Establish Azure cloud connection to a blob container 
    """
    sd =get_secrets()
    container_client = BlobServiceClient(
        account_url=sd['Azure_account_url'],
        credential=sd['Azure_account_key']
    ).get_container_client(container=sd['Azure_ml_blob_container'])
    return container_client

class FakeFile:
    def __init__(self, inbytes):
        self.f = io.BytesIO()
        self.f.write(inbytes)
        self.f.seek(0)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        self.f.close()

