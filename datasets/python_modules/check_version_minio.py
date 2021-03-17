import os
import boto3 

cloud_access_key         = "minio"
cloud_secret_key         = "miniostorage"
cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"

s3_client = boto3.client('s3',
                        aws_access_key_id = cloud_access_key,
                        aws_secret_access_key = cloud_secret_key,
                        endpoint_url= cloud_url, 
                        verify=False,
                        config=boto3.session.Config(signature_version='s3v4'))

s3_resource = boto3.resource('s3',
                        aws_access_key_id = cloud_access_key,
                        aws_secret_access_key = cloud_secret_key,
                        endpoint_url=cloud_url, 
                        verify=False,
                        config=boto3.session.Config(signature_version='s3v4'))

"""
This module can be used to check if a version created on minio does only include existing files. 
"""

dataset_name = "dataset-v1-dih4cps"
ds_short = "dsv1" 
dataset_version = "version_1_gray"
dataset_format = "voc"

file_list_path = f"downloads/{dataset_version}/{dataset_format}/{ds_short}_{dataset_version}_{dataset_format}.txt"
local_path = os.path.join(".", f"{ds_short}_{dataset_version}_{dataset_format}.txt")

bucket_name = dataset_name


def load_file_lists(path_to_file_list):
    with open(path_to_file_list, 'r') as f:
        content_in = f.readlines()

    content = {}
    key = ""
    for line in content_in:
        if line.count(":") > 0:
            key = line.split(":")[0]
            content[key] = []
            continue
        
        if not key == "":
            content[key].append(line.split("\n")[0])
    return content 

    
## download file list and load the list
s3_client.download_file(bucket_name, file_list_path, local_path)
file_list = load_file_lists(local_path)

def _key_existing_size__list(client, bucket, key):
    """return the key's size if it exist, else None"""
    response = client.list_objects_v2(
        Bucket=bucket,
        Prefix=key,
    )
    for obj in response.get('Contents', []):
        if obj['Key'] == key:
            return obj['Size']

## check if every file exists 
existing_files = []
missing_files = [] 
for d in ["train", "valid", "test"]:
    flist = file_list[d]
    for f in flist:
        size = _key_existing_size__list(s3_client, bucket_name, f)
        if size == None:
            missing_files.append(f)
        else:
            existing_files.append(f) 


print("Train:", len(file_list["train"])/2)
print("Valid:", len(file_list["valid"])/2)
print("Test:", len(file_list["test"])/2)
print(f"Existing files: {len(existing_files)}")
print(f"Missing files: {len(missing_files)}")

if len(missing_files) > 0:
    out_path = os.path.join(".", f"missing_files_{ds_short}_{dataset_version}_{dataset_format}.txt")
    with open(out_path, 'w') as out_file:
        for f in missing_files:
            out_file.write(f"{f}\n")
