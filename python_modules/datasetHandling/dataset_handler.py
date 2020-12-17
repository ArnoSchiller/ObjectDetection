import os
import sys
import csv
import glob
import boto3 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DATASET_PATH
from dataAugmentation.image_augmentation import ImageAugmentation
from helpers.xml_handler import XMLHandler

ACCESS_KEY      = "minio"
SECRET_KEY      = "miniostorage"
CLOUD_URL       = "https://minio.dih4cps.swms-cloud.com:9000/"


class DatasetHandler:

    data_transformation_path = os.path.join(os.path.dirname(__file__), "data_transformation")
    sys.path.append(data_transformation_path)

    download_dir = os.path.join(os.path.dirname(__file__), "downloads")
    temp_download_dir = os.path.join(os.path.dirname(__file__), "downloads_TEMP")

    eval_split = True
    train_test_split = True

    cloud_access_key         = ACCESS_KEY
    cloud_secret_key         = SECRET_KEY
    cloud_url                = CLOUD_URL

    bucket_names = ["dataset-v1-dih4cps", "dataset-v2-dih4cps"]
    video_bucket_name = "test-dih4cps"

    created_files = ["images.csv", "test.csv", "train.csv",
                     "images.record", "test.record", "train.record",
                     "eval.csv", "eval.record"]

    def __init__(self):
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

    def augmentate_dataset(self, dataset_name):
        """
        Download not augmentated images and augmentate them. Also duplicate the label file and upload both. 
        """
        dataset_part = "shrimp_detection"
        cloud_download_dir = "data/" + dataset_part + "/"
        #cloud_upload_dir = "data/" + dataset_part + "_aug" + "/"
        cloud_upload_dir = "data/" + dataset_part + "_gray" + "/"

        local_download_dir = os.path.join(os.path.dirname(__file__), "aug_temp")
        if not os.path.exists(local_download_dir):
            os.mkdir(local_download_dir)
        
        dataset_files = self.get_complete_dataset_list(dataset_name, filter_str=dataset_part+"/")
        print(len(dataset_files))

        #augmented_files = self.get_complete_dataset_list(dataset_name, filter_str="shrimp_detection_aug/")
        augmented_files = self.get_complete_dataset_list(dataset_name, filter_str="shrimp_detection_gray/")

        for aug_file in augmented_files:
            baseparts = aug_file.split("_")[0:-1]
            basename = ""
            for index, part in enumerate(baseparts):
                basename += part
                if index < len(baseparts) - 1:
                     basename += "_"
            if dataset_files.count(basename) > 0:
                dataset_files.remove(basename)
        print(len(dataset_files))

        img_aug = ImageAugmentation()

        for index, filename in enumerate(dataset_files):
            print("Augmentation file {} of {}".format(index, len(dataset_files)))
            
            ## download
            image_file_path = cloud_download_dir + filename + ".png"
            local_img_path = os.path.join(local_download_dir, filename + ".png")
            self.s3_client.download_file(dataset_name, image_file_path, local_img_path)

            xml_file_path = cloud_download_dir + filename + ".xml"
            local_xml_path = os.path.join(local_download_dir, filename + ".xml")
            self.s3_client.download_file(dataset_name, xml_file_path, local_xml_path)

            ## augmentation
            created_file_paths = img_aug.augmentate_image_file(local_img_path)

            ## upload and delete
            for file_path in created_file_paths:
                _, filename = os.path.split(file_path)
                cloud_path = cloud_upload_dir + filename
                self.s3_client.upload_file(file_path, dataset_name, cloud_path)

                os.remove(file_path)
        
            os.remove(local_xml_path)
            os.remove(local_img_path)

    def count_labels(self, csv_file_paths, output_tag):
        labels_dict = {}
        labels_dict["summary"] = {}
        labels_dict["summary"]["sum"] = 0 

        for file_path in csv_file_paths:
            file_name = os.path.split(file_path)[-1]
            labels_dict[file_name] = {}
            labels_dict[file_name]["sum"] = 0 
            # einlesen der csv file
            
            with open(file_path) as csv_file:
                csv_reader_object = csv.DictReader(csv_file)
                for row in csv_reader_object:
                    class_name = row["class"]

                    if not list(labels_dict[file_name].keys()).count(class_name) > 0:
                        labels_dict[file_name][class_name] = 0
                    if not list(labels_dict["summary"].keys()).count(class_name) > 0:
                        labels_dict["summary"][class_name] = 0

                    labels_dict[file_name][class_name] += 1 
                    labels_dict[file_name]["sum"] += 1 

                    labels_dict["summary"][class_name] += 1
                    labels_dict["summary"]["sum"] += 1

        print(labels_dict)


    def update_xml_files(self, dataset_name, changes=[], filter_str=""):
        
        xml_handler = XMLHandler()

        # cloud_download_dir = "data/shrimp_detection_aug/"
        cloud_download_dir = "data/shrimp_detection_gray/"

        local_download_dir = os.path.join(os.path.dirname(__file__), "TEMP_XML")
        if not os.path.exists(local_download_dir):
            os.mkdir(local_download_dir)
        
        dataset_files =  self.get_complete_dataset_list(dataset_name, filter_str=cloud_download_dir)
        # dataset_files = ['test1_cronjob_2020-09-30_12-37-56_100_brighter']
        for index, file_name in enumerate(dataset_files):
            print("Updating file {} of {}".format(index+1, len(dataset_files)))
            # download
            cloud_path = cloud_download_dir + file_name + ".xml"
            local_path = os.path.join(local_download_dir, file_name + ".xml")
            self.s3_client.download_file(dataset_name, cloud_path, local_path)

            # change xml file
            for change in changes:
                if change == "file_name":
                    xml_handler.change_xml_file(local_path, change, file_name + ".png")
                if change == "file_path":
                    path = cloud_download_dir + file_name + ".png"
                    xml_handler.change_xml_file(local_path, change, path)
                if change == "folder":
                    # folder = "shrimp_detection_aug"
                    folder = "shrimp_detection_gray"
                    xml_handler.change_xml_file(local_path, change, folder)

            # upload
            self.s3_client.upload_file(local_path, dataset_name, cloud_path)

            # delete
            os.remove(local_path)

    def upload_missing_images(self, bucket_name, image_dir_path):

        png_files = glob.glob(os.path.join(image_dir_path, "*.png"))

        for index, file_path in enumerate(png_files):
            print("File {} of {}.".format(index+1, len(png_files)))
            name = os.path.basename(file_path)
            name = name.split(".")[0]

            #image_path = os.path.join(image_dir_path, name + ".png")
            image_path = file_path

            if os.path.exists(image_path):
                """
                os.system("copy {} {}".format(image_path, os.path.join(self.download_dir, name + ".png")))
                os.remove(file_path)
                """
                self.s3_client.upload_file(image_path, bucket_name, "images/{}.png".format(name))

    def create_dataset_version(self, bucket_name, version_id, full_path=True):
        version_name = "version_{}".format(version_id)
        
        ## define dataset version, write every used image into txt
        self.define_dataset_version(bucket_name=bucket_name, version_name=version_name)

        ## load latest dataset as list 
        data_list = dsh.load_dataset_version(bucket_name=bucket_name, version_name=version_name)

        ## create download directory
        temp_dir = "TEMP_DOWNLOADS"
        labels_dir = temp_dir
        images_dir = temp_dir

        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        if full_path:
            for data_path in data_list:
                data_name = data_path.split("/")[-1]
                
                xml_path = data_path + ".xml"
                download_path = os.path.join(labels_dir, "{}.xml".format(data_name))
                self.s3_client.download_file(bucket_name, xml_path, download_path)

                png_path = data_path + ".png"
                download_path = os.path.join(images_dir, "{}.png".format(data_name))
                self.s3_client.download_file(bucket_name, png_path, download_path)
            #"""

        else:
            ## old version not needed anymore
            ## download every xml file and every png file
            for data_name in data_list:
                object_name = "labels/{}.xml".format(data_name)
                download_path = os.path.join(labels_dir, "{}.xml".format(data_name))
                self.s3_client.download_file(bucket_name, object_name, download_path)

                object_name = "images/{}.png".format(data_name)
                download_path = os.path.join(images_dir, "{}.png".format(data_name))
                self.s3_client.download_file(bucket_name, object_name, download_path)
        

        ## convert xml files to csv files
        command = "python xml_to_csv.py"
        args = ""
        if self.eval_split:
            args += " --splitted"
        elif self.train_test_split:
            args += " --train_test_split"
        args += " --xml_files_dir {}".format(labels_dir)
        print(command + args)
        os.system(command + args)

        ## convert csv files to tfrecord files 
        tags = ["images"]
        if self.eval_split:
            for i in ["eval", "test", "train"]:
                tags.append(i)
        elif self.train_test_split:
            for i in ["test", "train"]:
                tags.append(i)
        print(tags)
        for tag in tags:
            if os.path.exists(tag + ".csv"):  
                command = "python generate_tfrecord.py"
                args = ""
                args += " --csv_input={}.csv".format(tag)
                args += " --image_dir={}".format(images_dir)
                args += " --output_path={}.record".format(tag)
                print(command + args)
                os.system(command + args)
    	
        ## upload created files
        for file_name in self.created_files:
            local_path = os.path.join(os.path.dirname(__file__), file_name)
            if os.path.exists(local_path):
                cloud_path = version_name + "/" + file_name
                self.s3_client.upload_file(local_path, bucket_name, cloud_path)
        
        ## remove created files
        self.created_files.append("images.txt")
        for file_name in self.created_files:
            local_path = os.path.join(os.path.dirname(__file__), file_name)
            if os.path.exists(local_path):
                os.remove(local_path)

        ## remove downloaded files 
        """
        if os.path.exists(temp_dir):
            files = glob.glob(os.path.join(temp_dir, "*"))
            for file_path in files:
                os.remove(file_path)
            os.rmdir(temp_dir)
        #"""

    def define_dataset_version(self, bucket_name, version_name, full_path=True):
        ## define dataset version, write every used image into txt

        if not self.bucket_names.count(bucket_name) > 0:
            return
        images_file_name = "images.txt"
        bucket = self.s3_resource.Bucket(bucket_name)
        """
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count(version_name) > 0:
                print("Version ", version_name, " allready exists.")
                return
        """
        all_images = self.get_complete_dataset_list(bucket_name, full_path=full_path, filter_str="shrimp_detection/")
        print("Selected {} images to define version".format(len(all_images)))

        with open(images_file_name, "w") as out_file:
            for image_name in all_images:
                out_file.write(image_name + "\n")

        object_name = version_name + "/" + images_file_name
        self.s3_client.upload_file(images_file_name, bucket_name, object_name)

        os.remove(images_file_name)
    
    def get_complete_dataset_list(self, bucket_name, filter_str=None, full_path=False):

        if not self.bucket_names.count(bucket_name) > 0:
            return False

        dataset_files = []
        xml_file_names = []

        bucket = self.s3_resource.Bucket(bucket_name)
        
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count("xml") > 0:
                if filter_str is None or object_name.count(filter_str) > 0:
                    filepath = object_name.split(".")[0]
                    filename = filepath.split("/")[-1]
                    xml_file_names.append(filename)
                    
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count("png") > 0:
                if filter_str is None or object_name.count(filter_str) > 0:
                    filepath = object_name.split(".")[0]
                    filename = filepath.split("/")[-1]
                    if xml_file_names.count(filename) > 0:
                        if full_path:
                            dataset_files.append(filepath)
                        else:
                            dataset_files.append(filename)

        return dataset_files

    def download_evaluation_set(self, dataset_name, dataset_version, mode="csv", force_download=False):
        download_dir = os.path.join(DATASET_PATH, dataset_name, dataset_version)
        if not os.path.exists(download_dir) or force_download:
            os.mkdir(download_dir) 
        else:
            print("Directory allready exists: {}. Set force_download to True to overwrite files.".format(download_dir))
            labelmap_name = "labelmap.pbtxt"
            labelmap_path = os.path.join(download_dir, labelmap_name)
            images_path = os.path.join(download_dir, "eval_images")
            return labelmap_path, images_path, []
        
        # download labelmap
        labelmap_name = "labelmap.pbtxt"
        cloud_path = labelmap_name
        labelmap_path = os.path.join(download_dir, labelmap_name)
        self.s3_client.download_file(dataset_name, cloud_path, labelmap_path)


        if mode == "csv":
            csv_file = "eval.csv" 
            cloud_path = dataset_version + "/" + csv_file
            local_path = os.path.join(download_dir, csv_file)
            self.s3_client.download_file(dataset_name, cloud_path, local_path)

            eval_images = []
            with open(local_path) as csvfile:
                csv_reader_object = csv.DictReader(csvfile)

                for row in csv_reader_object:
                    file_name = row['filename']
                    if not eval_images.count(file_name) > 0:
                        eval_images.append(file_name)

            images_path = os.path.join(download_dir, "eval_images")
            if not os.path.exists(images_path):
                os.mkdir(images_path) 
            
            """
            bucket = self.s3_resource.Bucket(dataset_name)
            for bucket_object in bucket.objects.all():
                object_name = str(bucket_object.key)
                if object_name.count("data") > 0:
                    file_name = object_name.split("/")[-1]
                    if eval_images.count(file_name) > 0:
                        local_path = os.path.join(images_path, file_name)
                        self.s3_client.download_file(dataset_name, object_name, local_path)
            """

            for file_name in eval_images:
                cloud_path = "data/shrimp_detection/" + file_name
                local_path = os.path.join(images_path, file_name)
                self.s3_client.download_file(dataset_name, cloud_path, local_path)
            
            downloaded_images = glob.glob(os.path.join(download_dir, "*.png"))
            downloaded_images = [os.path.split(path)[-1] for path in downloaded_images]

            not_downloaded = set(eval_images).difference(downloaded_images)

            print("Downloaded {}/{} evaluation images".format(len(downloaded_images), len(eval_images)))

            return labelmap_path, images_path, list(not_downloaded)

    def download_not_labeled_images(self, bucket_name=None):
        
        if bucket_name is None:
            bucket_name = self.bucket_names[1]    

        if not os.path.exists(self.download_dir):
            os.mkdir(self.download_dir)

        image_files = self.get_all_image_names(bucket_name)
        label_files = self.get_all_label_names(bucket_name)

        for image_name in image_files:
            if label_files.count(image_name):
                continue
            image_file_path = "images/" + image_name + ".png"
            local_file_path = os.path.join(self.download_dir, image_name + ".png")
            self.s3_client.download_file(bucket_name, image_file_path, local_file_path)

    def download_missing_image_label(self, bucket_name=None):
        
        if bucket_name is None:
            bucket_name = self.bucket_names[1]    

        if not os.path.exists(self.download_dir):
            os.mkdir(self.download_dir)

        image_files = self.get_all_image_names(bucket_name)
        label_files = self.get_all_label_names(bucket_name)

        for label_name in label_files:
            if image_files.count(label_name):
                continue
            label_file_path = "labels/" + label_name + ".xml"
            local_file_path = os.path.join(self.download_dir, label_name + ".xml")
            self.s3_client.download_file(bucket_name, label_file_path, local_file_path)

    def download_video(self, object_name, download_path, bucket_name=None):

        if bucket_name is None:
            bucket_name = self.video_bucket_name

        self.s3_client.download_file(bucket_name, object_name, download_path)
    
    def load_dataset_version(self, bucket_name, version_name=None):
        
        images = []
        image_file_name = "images.txt"
        if version_name is None:
            version_name = self.get_latest_dataset_version(bucket_name)

        images_file = version_name + "/" + image_file_name
        self.s3_client.download_file(bucket_name, images_file, image_file_name)

        with open(image_file_name, "r") as in_file:
            for image_name in in_file:
                if image_name.count("\n"):
                    image_name = image_name.split("\n")[0]
                images.append(image_name)

        return images 

    def get_latest_dataset_version(self, bucket_name):
        return "version_2020-10-27"

    def get_all_video_names(self, filter_str="", full_path=False):
        video_file_names = []
        bucket = self.s3_resource.Bucket(self.video_bucket_name)
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count(".avi")>0 and object_name.count(filter_str)>0:
                if object_name.count("/") > 0:
                    object_name = object_name.split("/")[-1]
                filename = object_name.split(".")[0]

                if full_path:
                    video_file_names.append(object_name.split(".")[0])
                else:
                    video_file_names.append(filename)
        return video_file_names

    def get_all_image_names(self, bucket_name, full_path=False):

        if not self.bucket_names.count(bucket_name) > 0:
            return False

        png_file_names = []

        bucket = self.s3_resource.Bucket(bucket_name)
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count("png") > 0:
                filepath = object_name.split(".")[0]
                filename = filepath.split("/")[-1]
                
                if full_path:
                    png_file_names.append(object_name.split(".")[0])
                else:
                    png_file_names.append(filename)
        return png_file_names

    def get_all_label_names(self, bucket_name):

        if not self.bucket_names.count(bucket_name) > 0:
            return False

        xml_file_names = []

        bucket = self.s3_resource.Bucket(bucket_name)
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count("xml") > 0:
                filepath = object_name.split(".")[0]
                filename = filepath.split("/")[-1]
                xml_file_names.append(filename)
        return xml_file_names

    def create_dataset_config(self, dataset_name):
        import xml.etree.ElementTree as ET

        ## Setup config xml tree
        dataset_xml = ET.Element('dataset')
        ds_name = ET.SubElement(dataset_xml, "name")
        ds_name.text = dataset_name

        ## Include classes to xml tree
        # download labelmap 
        if not os.path.exists(self.temp_download_dir):
            os.mkdir(self.temp_download_dir)
        label_map_file = "labelmap.pbtxt"
        local_path = os.path.join(self.temp_download_dir, label_map_file)
        
        self.s3_client.download_file(dataset_name, 
                label_map_file, 
                local_path)    
        classes = self.get_classes_from_labelmap(local_path)
        os.remove(local_path)
        
        # add classes to xml tree
        ds_classes = ET.SubElement(dataset_xml, "classes")
        for label_class in classes:
            ds_class = ET.SubElement(ds_classes, "name")
            ds_class.text = label_class
        
        # add versions with description
        versions = self.get_possible_versions(dataset_name)
        for version in versions:
            ds_version = ET.SubElement(dataset_xml, "version")
            ds_vtag = ET.SubElement(ds_version, "tag")
            ds_vtag.text = version

            ds_vimg = ET.SubElement(ds_version, "num_images")
            ds_vimg.text = self.get_num_txt(dataset_name, version, "images")

            ds_vtrn = ET.SubElement(ds_version, "num_train_images")
            ds_vtrn.text = self.get_num_csv(dataset_name, version, "train")

            ds_vval = ET.SubElement(ds_version, "num_eval_images")
            ds_vval.text = self.get_num_csv(dataset_name, version, "eval")

            ds_vtst = ET.SubElement(ds_version, "num_test_images")
            ds_vtst.text = self.get_num_csv(dataset_name, version, "test")

        # save config file and upload it
        output_file_name = "dataset_config.xml"
        with open(output_file_name, "wb") as output_file:
            output_file.write(self.prettify(dataset_xml))

        self.s3_client.upload_file(output_file_name,    # local filepath
                dataset_name,                           # bucket name
                output_file_name)                       # cloud filepath
        os.remove(output_file_name)
        os.rmdir(self.temp_download_dir)
    
    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        import xml.etree.ElementTree as ET
        from lxml import etree
        rough_string = ET.tostring(elem, 'utf8').replace("\t".encode(), "  ".encode())
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True,
                                    xml_declaration=True,
                                    encoding='UTF-8')
    
    def get_classes_from_labelmap(self, file_path):
        classes = []
        with open(file_path, "r") as input_file:
            for line in input_file.readlines():
                if line.count('name') > 0:
                    classes.append(line.split("'")[-2])
        return classes

    def get_possible_versions(self, dataset_name):
        bucket = self.s3_resource.Bucket(dataset_name)
        versions = []
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count("/") > 0:
                tag = object_name.split("/")[0]
                if not tag == "data" and versions.count(tag) == 0:
                    versions.append(tag)
        return versions

    def get_num_txt(self, dataset_name, dataset_version, file_name):
        file_name = file_name + ".txt"
        local_file = os.path.join(self.temp_download_dir, file_name)
        try: 
            self.s3_client.download_file(dataset_name, 
                dataset_version + '/' + file_name, 
                local_file)
        except:
            print("{} not found.".format(dataset_version + '/' + file_name))
            return "None"
        
        with open(local_file, 'r') as txt_file:
            file_str = txt_file.readlines()
        os.remove(local_file)

        return str(len(file_str))

    def get_num_csv(self, dataset_name, dataset_version, file_name):
        file_name = file_name + ".csv"
        local_file = os.path.join(self.temp_download_dir, file_name)
        try: 
            self.s3_client.download_file(dataset_name, 
                dataset_version + '/' + file_name, 
                local_file)
        except:
            print("{} not found.".format(dataset_version + '/' + file_name))
            return "None"

        import pandas as pd
        df = pd.read_csv(local_file)

        os.remove(local_file)
        return str(len(df.index))

if __name__ == "__main__":
    dsh = DatasetHandler()

    #dsh.download_not_labeled_images()
    #dsh.download_missing_image_label()
    #dsh.upload_missing_images(bucket_name=dsh.bucket_names[0], image_dir_path = dsh.download_dir)# image_dir_path= os.path.abspath("C:/Users/Schiller/Downloads/images_detected"))
    
    #dsh.create_dataset_version(dsh.bucket_names[0], "2020-11-09")
    #dsh.create_dataset_config(dsh.bucket_names[0])
    
    
    #dsh.create_dataset_version(dsh.bucket_names[1],"2020-12-01")
    #dsh.create_dataset_config(dsh.bucket_names[0])
    
    #dsh.download_evaluation_set(dsh.bucket_names[0],"version_2020-12-01")

    #dsh.augmentate_dataset(dsh.bucket_names[1])

    #dsh.update_xml_files(dsh.bucket_names[1], changes=["file_name", "file_path", "folder"])

    dir_path = os.path.join(os.path.dirname(__file__), "TEMP_COUNT")
    # paths = [os.path.join(dir_path, "train.csv"), os.path.join(dir_path, "test.csv")]
    paths = [os.path.join(dir_path, "train.csv"), os.path.join(dir_path, "test.csv"), os.path.join(dir_path, "eval.csv")]
    dsh.count_labels(paths, "test_log")