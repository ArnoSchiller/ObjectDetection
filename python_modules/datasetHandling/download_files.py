import boto3
import os

date_str = "2020-10-06"
download_dir = os.path.join(os.path.abspath("./video_files"), date_str)
if not os.path.exists(download_dir):
    os.mkdir(download_dir)

cloud_access_key         = "minio"
cloud_secret_key         = "miniostorage"
cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"
# cloud_bucket_name        = "test-dih4cps"
cloud_bucket_name        = "dataset-v2-dih4cps"

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

"""
# download video files
for bucket_object in s3_bucket.objects.all():
    object_name = str(bucket_object.key)
    if object_name.count(date_str) > 0:
        s3_client.download_file(cloud_bucket_name, bucket_object.key, os.path.join(download_dir, bucket_object.key))

#""" 

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
base_dir = os.path.join(os.getcwd(), "images")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
proof_dir(paths=[base_dir, train_dir, test_dir])

test_percentage = 0.2

num_test_files = len(dataset_files) * test_percentage
counter_train_files = 0
counter_test_files = 0

download_dir = test_dir

for file_name in dataset_files:
    xml_file_name = "labels/" + file_name + ".xml"
    print(xml_file_name)
    s3_client.download_file(cloud_bucket_name, xml_file_name, os.path.join(test_dir, xml_file_name))
    png_file_name = "images/" + file_name + ".png"
    s3_client.download_file(cloud_bucket_name, png_file_name, os.path.join(test_dir, png_file_name))
    
    if counter_test_files >= num_test_files:
        download_dir = train_dir
        counter_train_files += 1
    else:
        counter_test_files += 1

percentage = int(100 * counter_test_files / (counter_test_files + counter_train_files))
print("Downloaded ", counter_train_files, " train and ", counter_test_files, " test files. Test percentage: ", percentage) 
