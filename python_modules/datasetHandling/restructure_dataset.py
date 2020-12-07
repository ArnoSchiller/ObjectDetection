import os
import boto3 

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

from dataset_handler import DatasetHandler 

download_dir = os.path.join(os.path.dirname(__file__), "TEMP_XML")
if not os.path.exists(download_dir):
    os.mkdir(download_dir)

cloud_access_key         = "minio"
cloud_secret_key         = "miniostorage"
cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"

s3_client = boto3.client('s3',
                        aws_access_key_id = cloud_access_key,
                        aws_secret_access_key = cloud_secret_key,
                        endpoint_url= cloud_url, 
                        verify=True,
                        config=boto3.session.Config(signature_version='s3v4'))

s3_resource = boto3.resource('s3',
                        aws_access_key_id = cloud_access_key,
                        aws_secret_access_key = cloud_secret_key,
                        endpoint_url=cloud_url, 
                        verify=True,
                        config=boto3.session.Config(signature_version='s3v4'))

data_v1_bucket_name = "dataset-v1-dih4cps"
data_v2_bucket_name = "dataset-v2-dih4cps"

bucket_name = data_v1_bucket_name
bucket = s3_resource.Bucket(bucket_name)
        
files_list = []
for bucket_object in bucket.objects.all():
    object_name = str(bucket_object.key)
    if object_name.count("xml") > 0:
        filepath = object_name.split(".")[0]
        if filepath.count("labels/") > 0:
            filename = filepath.split("/")[-1]
            files_list.append(filename)
        
for file_name in files_list:
    xml_file_path = "labels/" + file_name + ".xml"
    png_file_path = "images/" + file_name + ".png"
    
    local_file_path = os.path.join(download_dir, file_name + ".xml")

    # download xml file 
    s3_client.download_file(bucket_name, xml_file_path, local_file_path)

    # setup xml parser
    parser = etree.XMLParser()
    xmltree = ElementTree.parse(local_file_path, parser=parser).getroot()
    
    # proof if the annotations includes objects
    objects = xmltree.findall('object')
    if len(objects) <= 0:
        print(file_name)
        base_path = "data/no_detection/{}".format(file_name)
    else:
        base_path = "data/shrimp_detection/{}".format(file_name)

    # copy png file 
    #print(base_path)
    s3_resource.Object(bucket_name, base_path + ".png").copy_from(CopySource=bucket_name + "/" + png_file_path)
    s3_resource.Object(bucket_name, png_file_path).delete()
    s3_resource.Object(bucket_name, base_path + ".xml").copy_from(CopySource=bucket_name + "/" + xml_file_path)
    s3_resource.Object(bucket_name, xml_file_path).delete()
    
    # delete local files 
    # os.remove(local_file_path)
    