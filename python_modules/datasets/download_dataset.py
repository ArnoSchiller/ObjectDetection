import boto3
import os


def download_dataset_tf_record(dataset_name, dataset_version):
    """
    Dowmload dataset using tensorflow record format. 
    The train-test and training-validation percentage is already defined (see dataset config). 
    """
    base_path = os.path.dirname(__file__)
    while not os.path.split(base_path)[-1] == "ObjectDetection":
        base_path = os.path.dirname(base_path)

    datasets_storage_path = os.path.join(base_path, "datasets")

    download_path = os.path.join(datasets_storage_path, dataset_name)
    if not os.path.exists(download_path):
        os.mkdir(download_path)
    download_path = os.path.join(download_path, dataset_version)
    if not os.path.exists(download_path):
        os.mkdir(download_path)

    cloud_access_key         = "minio"
    cloud_secret_key         = "miniostorage"
    cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"
    cloud_bucket_name        = dataset_name

    s3_client = boto3.client('s3',
                    aws_access_key_id = cloud_access_key,
                    aws_secret_access_key = cloud_secret_key,
                    endpoint_url=cloud_url, 
                    verify=True,
                    config=boto3.session.Config(signature_version='s3v4'))

    files_to_download = ['train.record', 'eval.record', 'test.record']
    for file_name in files_to_download:
        s3_client.download_file(cloud_bucket_name, 
                                dataset_version + '/' + file_name, 
                                os.path.join(download_path, file_name))     


def download_dataset_images(dataset_name, with_no_detection=True):
    base_path = os.path.dirname(__file__)
    while not os.path.split(base_path)[-1] == "ObjectDetection":
        base_path = os.path.dirname(base_path)

    datasets_storage_path = os.path.join(base_path, "datasets")

    download_path = os.path.join(datasets_storage_path, dataset_name)
    if not os.path.exists(download_path):
        os.mkdir(download_path)
    download_path = os.path.join(download_path, "data")
    if not os.path.exists(download_path):
        os.mkdir(download_path)


    cloud_access_key         = "minio"
    cloud_secret_key         = "miniostorage"
    cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"
    cloud_bucket_name        = dataset_name

    s3_client = boto3.client('s3',
                    aws_access_key_id = cloud_access_key,
                    aws_secret_access_key = cloud_secret_key,
                    endpoint_url=cloud_url, 
                    verify=True,
                    config=boto3.session.Config(signature_version='s3v4'))

    s3_resource = boto3.resource('s3',
                    aws_access_key_id = cloud_access_key,
                    aws_secret_access_key = cloud_secret_key,
                    endpoint_url=cloud_url, 
                    verify=True,
                    config=boto3.session.Config(signature_version='s3v4'))

    bucket = s3_resource.Bucket(cloud_bucket_name)
        
    for bucket_object in bucket.objects.all():
        object_name = str(bucket_object.key)
        if object_name.count("data") > 0:
            if object_name.count("no_detection") > 0 and not with_no_detection:
                continue
            if object_name.count("xml") > 0 or object_name.count("png") > 0: 
                filedir = object_name.split("/")[-2]
                filename = object_name.split("/")[-1]
                if not os.path.exists(os.path.join(download_path, filedir)):
                    os.mkdir(os.path.join(download_path, filedir))
                s3_client.download_file(cloud_bucket_name, 
                        object_name, 
                        os.path.join(download_path, filedir, filename))     

download_dataset_tf_record(dataset_name="dataset-v1-dih4cps", dataset_version="version_2020-12-01")
#download_dataset_images(dataset_name="dataset-v1-dih4cps")


"""

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
"""
"""
for file_name in dataset_files:
    """"""
    xml_file_name = file_name + ".xml"
    print(xml_file_name)
    s3_client.download_file(cloud_bucket_name, "labels/" + xml_file_name, os.path.join(download_dir, xml_file_name))
    #""""""
    png_file_name = file_name + ".png"
    s3_client.download_file(cloud_bucket_name, "images/" + png_file_name, os.path.join(download_dir, png_file_name))
    
    if counter_test_files >= num_test_files:
        download_dir = train_dir
        counter_train_files += 1
    else:
        counter_test_files += 1

percentage = int(100 * counter_test_files / (counter_test_files + counter_train_files))
print("Downloaded ", counter_train_files, " train and ", counter_test_files, " test files. Test percentage: ", percentage) 
"""