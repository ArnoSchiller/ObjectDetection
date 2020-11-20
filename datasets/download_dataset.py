import boto3
import os

image_dir               = "images"
label_v1_dir            = "labels_v1"
label_v2_dir            = "labels_v2"

cloud_access_key         = "minio"
cloud_secret_key         = "miniostorage"
cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"
# cloud_bucket_name        = "test-dih4cps"
cloud_bucket_name        = "dataset-v1-dih4cps" # "dataset-v2-dih4cps"

s3_client = boto3.client('s3',
                aws_access_key_id = cloud_access_key,
                aws_secret_access_key = cloud_secret_key,
                endpoint_url=cloud_url, 
                verify=False,
                config=boto3.session.Config(signature_version='s3v4'))
                
s3_resource = boto3.resource('s3',
                aws_access_key_id = cloud_access_key,
                aws_secret_access_key = cloud_secret_key,
                endpoint_url=cloud_url, 
                verify=False,
                config=boto3.session.Config(signature_version='s3v4'))

s3_bucket = s3_resource.Bucket(cloud_bucket_name) 

# download dataset (test / train splitted)
# get a list of every xml file also has an image
dataset_files = []
xml_file_names = []
for bucket_object in s3_bucket.objects.all():
    object_name = str(bucket_object.key)
    if object_name.count("xml") > 0:
        filepath = object_name.split(".")[0]
        filename = filepath.split("/")[1]
        xml_file_names.append(filename)
        
for bucket_object in s3_bucket.objects.all():
    object_name = str(bucket_object.key)
    if object_name.count("png") > 0:
        filepath = object_name.split(".")[0]
        filename = filepath.split("/")[1]
        if xml_file_names.count(filename) > 0:
            dataset_files.append(filename)

# proof directory paths
def proof_dir(paths = None):
    if len(paths) < 1:
        return False
    for path in paths:
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)

# download selected files doing train/test split
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
proof_dir(paths=[base_dir])

test_percentage = 0

if test_percentage > 0 and test_percentage < 1:
    download_dir = test_dir
    num_test_files = len(dataset_files) * test_percentage
    proof_dir(paths=[train_dir, test_dir])
else:
    download_dir = base_dir
    num_test_files = len(dataset_files)

counter_train_files = 0
counter_test_files = 0


for file_name in dataset_files:
    #"""
    xml_file_name = file_name + ".xml"
    print(xml_file_name)
    s3_client.download_file(cloud_bucket_name, "labels/" + xml_file_name, os.path.join(download_dir, xml_file_name))
    #"""
    png_file_name = file_name + ".png"
    s3_client.download_file(cloud_bucket_name, "images/" + png_file_name, os.path.join(download_dir, png_file_name))
    
    if counter_test_files >= num_test_files:
        download_dir = train_dir
        counter_train_files += 1
    else:
        counter_test_files += 1

percentage = int(100 * counter_test_files / (counter_test_files + counter_train_files))
print("Downloaded ", counter_train_files, " train and ", counter_test_files, " test files. Test percentage: ", percentage) 
