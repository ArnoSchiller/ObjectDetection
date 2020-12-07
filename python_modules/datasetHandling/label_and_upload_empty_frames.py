import os, glob
import cv2
import boto3

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

ACCESS_KEY      = "minio"
SECRET_KEY      = "miniostorage"
CLOUD_URL       = "https://minio.dih4cps.swms-cloud.com:9000/"

class EmptyFrameLabeling():
    """
    This class is used to convert video files to images. If a frame is allready labeld, the frame will not be saved as an image. 
    """        
    
    files_dir        = os.path.join(os.path.dirname(__file__), "empty_frames")
    basic_label_path = os.path.join(os.path.dirname(__file__), "basic_label.xml")

    cloud_access_key         = ACCESS_KEY
    cloud_secret_key         = SECRET_KEY
    cloud_url                = CLOUD_URL

    label_bucket_name        = "dataset-v2-dih4cps"
    cloud_base_path          = "data/no_detection/" 
    
    def __init__(self):

        if not os.path.exists(self.files_dir):
            quit()

        # setup the cloud connection 
        self.s3_client = boto3.client('s3',
                        aws_access_key_id = self.cloud_access_key,
                        aws_secret_access_key = self.cloud_secret_key,
                        endpoint_url=self.cloud_url, 
                        verify=False,
                        config=boto3.session.Config(signature_version='s3v4'))

        # get all files to process
        self.image_files = glob.glob(os.path.join(self.files_dir, "*.png"))

    def label_empty_images(self):

        for file_path in self.image_files:
            file_name = os.path.basename(file_path)

            parser = etree.XMLParser()
            xmltree = ElementTree.parse(self.basic_label_path, parser=parser).getroot()
            
            ## changing the filename and the path 
            file_name_label = xmltree.find('filename')
            file_name_label.text = file_name
            file_path_label = xmltree.find('path')
            file_path_label.text = self.cloud_base_path + file_name

            ## save the file 
            label_file_path = os.path.join(self.files_dir, file_name.split(".")[0] + ".xml")
            self.save(xmltree, label_file_path)

            ## upload xml and png file 
            cloud_path = self.cloud_base_path + file_name
            self.s3_client.upload_file(file_path, self.label_bucket_name, cloud_path)
            cloud_path = cloud_path.split(".")[0] + ".xml"
            self.s3_client.upload_file(label_file_path, self.label_bucket_name, cloud_path)

            ## remove local xml and png file 
            os.remove(file_path)
            os.remove(label_file_path)
            
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


if __name__ == "__main__":
    efl = EmptyFrameLabeling()
    efl.label_empty_images()