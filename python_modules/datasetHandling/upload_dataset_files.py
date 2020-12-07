import glob
import os

from dataset_handler import DatasetHandler

class FileUploader:

    def __init__(self, dataset_name=None):
        self.dataset_handler = DatasetHandler()

        if not dataset_name is None:
            self.bucket = dataset_name
        else:
            self.bucket = self.dataset_handler.bucket_names[1]  # 1 --> v2

    def upload_files_to_dataset(self, files_dir, file_type):
        """
        files_dir       directory of local files 
        file_type       png or xml
        upload_dir      directory to upload files to 
        """
        uploaded_files = []
        if file_type == "png":
            uploaded_files = self.dataset_handler.get_all_image_names(self.bucket)
            upload_dir = "images"
        elif file_type == "xml":
            uploaded_files = self.dataset_handler.get_all_label_names(self.bucket)
            upload_dir = "labels"
        else:
            return False

        filter_str = os.path.join(files_dir, "*." + file_type)
        file_paths = glob.glob(filter_str)

        upload_counter = 0 

        for path in file_paths:
            file_name = os.path.basename(path).split(".")[0]
    
            if uploaded_files.count(file_name) > 0:
                continue 

            upload_file_path = upload_dir + "/" + file_name + "." + file_type

            upload_counter += 1
            
            self.dataset_handler.s3_client.upload_file(path, self.bucket, upload_file_path)

        print("Uploaded ", upload_counter, " ", file_type, " files")

if __name__ == "__main__":
    image_dir = os.path.abspath("C:/Users/swms-hit/Schiller/DIH4CPS-PYTESTS/videoPreprocessing/video_files/Images")
    labels_v2_dir = os.path.abspath("C:/Users/swms-hit/Schiller/DIH4CPS-PYTESTS/videoPreprocessing/video_files/labels_v2")
    label_img_dir = os.path.abspath("C:/Users/swms-hit/Schiller/LabelImg/labels")

    fu = FileUploader()
    # fu.upload_files_to_dataset(image_dir, "png")
    fu.upload_files_to_dataset(label_img_dir, "xml")
    