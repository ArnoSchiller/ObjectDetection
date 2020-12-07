"""
This script can be used to copy and convert xml files from the dataset v2 to dataset v1 by changing the classesnames to shrimp for every class. 

Make sure you have installed the following packages:
    - pip install beautifulsoup4
"""

import os
import glob
import boto3 

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

from dataset_handler import DatasetHandler 
from dataset_handler import ACCESS_KEY, SECRET_KEY, CLOUD_URL

FILE_NAME_ADDITION = ""

class CloudFilesConverter:
    download_dir = os.path.dirname(__file__)

    cloud_access_key         = "minio"
    cloud_secret_key         = "miniostorage"
    cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"

    def __init__(self):

        self.dataset_handler = DatasetHandler()

        self.cvt = Converter()
        
        self.s3_client = boto3.client('s3',
                        aws_access_key_id = self.cloud_access_key,
                        aws_secret_access_key = self.cloud_secret_key,
                        endpoint_url=self.cloud_url, 
                        verify=True,
                        config=boto3.session.Config(signature_version='s3v4'))
                        
        self.s3_resource = boto3.resource('s3',
                        aws_access_key_id = self.cloud_access_key,
                        aws_secret_access_key = self.cloud_secret_key,
                        endpoint_url=self.cloud_url, 
                        verify=True,
                        config=boto3.session.Config(signature_version='s3v4'))

        self.data_v1_bucket_name = self.dataset_handler.bucket_names[0]
        self.data_v2_bucket_name = self.dataset_handler.bucket_names[1]

        self.s3_bucket_v1 = self.s3_resource.Bucket(self.data_v1_bucket_name)
        self.s3_bucket_v2 = self.s3_resource.Bucket(self.data_v2_bucket_name)

    def change_every_xml(self):
        files_list = self.get_data_to_convert()
        
        for file_path in files_list:

            xml_file_path = file_path + ".xml"
            png_file_path = file_path + ".png"
            
            # if the file is listed in data/no_detection no modification is needed, copy both files and continue
            if file_path.count("no_detection") > 0:
                self.s3_resource.Object(self.data_v1_bucket_name, png_file_path).copy_from(CopySource=self.data_v2_bucket_name + "/" + png_file_path)
                self.s3_resource.Object(self.data_v1_bucket_name, xml_file_path).copy_from(CopySource=self.data_v2_bucket_name + "/" + xml_file_path)
                continue

            file_name = file_path.split("/")[-1]

            local_file_path = os.path.join(self.download_dir, file_name + ".xml")

            new_xml_file_path = file_path + FILE_NAME_ADDITION + ".xml"

            # download xml file 
            self.s3_client.download_file(self.data_v2_bucket_name, xml_file_path, local_file_path)

            # change classes name for every object
            self.cvt.change_single_xml(path=local_file_path)

            # upload new xml file
            self.s3_client.upload_file(local_file_path, self.data_v1_bucket_name, new_xml_file_path)

            # copy png file 
            self.s3_resource.Object(self.data_v1_bucket_name, png_file_path).copy_from(CopySource=self.data_v2_bucket_name + "/" + png_file_path)

            # delete local files 
            os.remove(local_file_path)
            if not FILE_NAME_ADDITION == "":
                new_local_file_path = local_file_path.split(".")[0] + FILE_NAME_ADDITION + ".xml"
                os.remove(new_local_file_path)
            

    def get_data_to_convert(self):
        dataset_v2_files = self.dataset_handler.get_complete_dataset_list(self.data_v2_bucket_name, full_path=True)
        dataset_v1_files = self.dataset_handler.get_complete_dataset_list(self.data_v1_bucket_name)

        print("v2: ", len(dataset_v2_files))
        print("v1: ", len(dataset_v1_files))

        files_to_convert = []
        for filepath in dataset_v2_files:
            filename = filepath.split("/")[-1]
            if not dataset_v1_files.count(filename + FILE_NAME_ADDITION) > 0:
                files_to_convert.append(filepath)

        print("converts: ", len(files_to_convert))

        return files_to_convert

        

class Converter:

    def __init__(self):
        dir_path = os.path.dirname(__file__)
        filter_str = os.path.join(dir_path, "*.xml")
        self.file_paths = glob.glob(filter_str)

    def change_every_xml(self):
        """
        changes every xml file in directory.
        """
        for file_path in self.file_paths:
            self.change_single_xml(file_path)
            

    def change_single_xml(self, path):
        if not FILE_NAME_ADDITION == "" and path.count(FILE_NAME_ADDITION+".xml") > 0:
            print("Allready changed file ", path)
            return

        if not os.path.exists(path):
            return 

        file_path = path
        new_file_path = file_path.split(".")[0] + FILE_NAME_ADDITION + ".xml"

        parser = etree.XMLParser()
        xmltree = ElementTree.parse(file_path, parser=parser).getroot()
        
        ## changing the name of the annotated class for every object
        for object_iter in xmltree.findall('object'):
            name = object_iter.find('name')
            name.text = 'shrimp'

        ## save the file 
        self.save(xmltree, new_file_path)

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True).replace("  ".encode(), "\t".encode())

    def save(self, root, targetFile):
        out_file = None
        
        out_file = codecs.open(targetFile, 'w')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

    # """

if __name__ == "__main__":

    cvt = CloudFilesConverter()
    cvt.change_every_xml()