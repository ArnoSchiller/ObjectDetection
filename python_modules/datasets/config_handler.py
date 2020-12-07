from IPython.display import display 
import xml.etree.ElementTree as ET 
import pandas as pd 
import os
import boto3 

ACCESS_KEY      = "minio"
SECRET_KEY      = "miniostorage"
CLOUD_URL       = "https://minio.dih4cps.swms-cloud.com:9000/"

base_path = os.path.dirname(__file__)
while not os.path.split(base_path)[-1] == "ObjectDetection":
    base_path = os.path.dirname(base_path)

datasets_storage_path = os.path.join(base_path, "datasets")
print(datasets_storage_path)

class DatasetConfigHandler():

    def __init__(self, dataset_name):
        # set paths
        self.dataset_path   = os.path.join(datasets_storage_path, dataset_name)
        conf_file_name      = "dataset_config.xml"
        conf_file_path      = os.path.join(self.dataset_path, conf_file_name)

        # update config file 
        try:
            s3_client = boto3.client('s3',
                        aws_access_key_id = ACCESS_KEY,
                        aws_secret_access_key = SECRET_KEY,
                        endpoint_url=CLOUD_URL, 
                        verify=True,
                        config=boto3.session.Config(signature_version='s3v4'))
            s3_client.download_file(dataset_name, 
                conf_file_name, 
                conf_file_path) 
            print("[INFO] Updated config file for {}.".format(dataset_name))

        except: 
            if not os.path.exists(conf_file_path):
                print("[ERROR] Cannot find config file")
                quit()

        # parse xml file        
        tree = ET.parse(conf_file_path)
        self.root = tree.getroot()

    def get_dataset_name(self):
        return self.root.find('name').text

    def get_dataset_classes(self):
        classes = []
        for element in self.root.findall('classes/name'):
            classes.append(element.text)
        return classes

    def get_dataset_versions(self):
        versions = []
        for element in self.root.findall('version'):
            versions.append(element.find('tag').text)
        return versions

    def get_version_details(self, version_tag):
        informations = {}
        fields = ['num_images', 'num_train_images', 'num_val_images', 'num_test_images']
        for element in self.root.findall('version'):
            if element.find('tag').text == version_tag:
                for field in fields:
                    informations[field] = element.find(field).text
        return informations

    def get_versions_overview(self):
        version_names       = []
        num_images          = []
        num_train_images    = []
        num_val_images      = []   
        num_test_images     = []

        for version in self.get_dataset_versions():
            version_names.append(version)

            details = self.get_version_details(version)
            num_images.append(details['num_images'])
            num_train_images.append(details['num_train_images'])
            num_val_images.append(details['num_val_images'])
            num_test_images.append(details['num_test_images'])

        overview = {
            'name' : version_names,
            'total num images' : num_images,
            'num train images' : num_train_images,     
            'num val images' : num_val_images,   
            'num test images' : num_test_images, 
        }
        return overview

    def summery_dataset(self):
        print('+','-'*78,'+')
        print("Dataset name:      {}".format(self.get_dataset_name()))
        print("Included classes:  {}".format(self.get_dataset_classes()))
        print("Version overview:")
        df = pd.DataFrame(self.get_versions_overview()) 
        display(df) 
        print('+','-'*78,'+')
