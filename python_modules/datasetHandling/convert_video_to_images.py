import os, glob
import cv2
import boto3

ACCESS_KEY      = "minio"
SECRET_KEY      = "miniostorage"
CLOUD_URL       = "https://minio.dih4cps.swms-cloud.com:9000/"

class VideoConverter():
    """
    This class is used to convert video files to images. If a frame is allready labeld, the frame will not be saved as an image. 
    """        
    
    download_dir    = os.path.join(os.path.dirname(__file__), "downloads_convert")
    log_path        = os.path.join(os.path.dirname(__file__), "log_convert.txt")

    cloud_access_key         = ACCESS_KEY
    cloud_secret_key         = SECRET_KEY
    cloud_url                = CLOUD_URL

    video_bucket_name        = "test-dih4cps"
    label_bucket_name        = "dataset-v2-dih4cps"

    capture = None
    date_filter = "2020-10-01"
    
    def __init__(self):

        self.video_index = -1
        self.frame_index = -1

        self.video_files = []
        self.label_files = []

        if not os.path.exists(self.download_dir):
            os.mkdir(self.download_dir)

        # setup the cloud connection 
        self.s3_client = boto3.client('s3',
                        aws_access_key_id = self.cloud_access_key,
                        aws_secret_access_key = self.cloud_secret_key,
                        endpoint_url=self.cloud_url, 
                        verify=False,
                        config=boto3.session.Config(signature_version='s3v4'))

        self.s3_resource = boto3.resource('s3',
                        aws_access_key_id = self.cloud_access_key,
                        aws_secret_access_key = self.cloud_secret_key,
                        endpoint_url=self.cloud_url, 
                        verify=False,
                        config=boto3.session.Config(signature_version='s3v4'))

        bucket = self.s3_resource.Bucket(self.video_bucket_name)

        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count(self.date_filter) > 0 and object_name.count("avi") > 0:
                filepath = object_name.split(".")[0]
                filename = filepath.split("/")[-1]
                self.video_files.append(filename)  
        
        # read log and remove every seen video file 
        log_videos = self.read_logs()
        #self.video_files = ["test1_cronjob_2020-09-30_14-33-43"]
        for video_file in log_videos:
            if self.video_files.count(video_file) > 0:    
                self.video_files.remove(video_file)

        if len(self.video_files) <= 0:
            print("No videos to convert.")
            quit()

        self.log_file_writer = open(self.log_path, 'a')

    def convert_videos(self):

        while True:
            frame = self.grab_frame()
            frame_name = self.video_files[self.video_index] + "_{}".format(self.frame_index)

            # if the frame is not labeled yet, save it
            if not self.label_files.count(frame_name) > 0:
                image_path = os.path.join(self.download_dir, frame_name + ".png")
                cv2.imwrite(image_path, frame)

    def download_video(self):

        video_name = self.video_files[self.video_index]
        cloud_path = video_name + ".avi"
        local_path = video_name + ".avi"
        self.s3_client.download_file(self.video_bucket_name, cloud_path, local_path)

    def change_capture(self):
        
        if self.video_index > -1:
            # add last file to log
            log_str = "{},{}\n".format(self.video_files[self.video_index], self.frame_index)
            self.log_file_writer.write(log_str)

            # remove last downloaded avi file 
            self.capture.release()
            file_path = os.path.abspath(self.video_files[self.video_index] + ".avi")
            os.remove(file_path)

        # reset index
        self.video_index += 1
        self.frame_index = 0 

        # if every video was seen quit
        if self.video_index >= len(self.video_files):
            self.log_file_writer.close()
            quit()

        # download current video and create video capture
        self.download_video()
        video_name = self.video_files[self.video_index]
        video_path = os.path.abspath(video_name + ".avi") 
        self.capture = cv2.VideoCapture(video_path)

        # get every labeled frame from this video file 
        self.label_files = self.get_labeled_frames(video_name)
        print(self.label_files)

        print("Video file {} of {}".format(self.video_index + 1, len(self.video_files)))


    def grab_frame(self):

        if self.capture is None or not self.capture.isOpened():
            self.change_capture()
        frame = None
        while frame is None:
            ret,frame = self.capture.read()
            self.frame_index += 1
            if not ret:
                self.change_capture()
        return frame

    def get_labeled_frames(self, video_name):
        label_files = []
        bucket = self.s3_resource.Bucket(self.label_bucket_name)
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count(video_name)>0 and object_name.count(".xml")>0:
                filepath = object_name.split(".")[0]
                filename = filepath.split("/")[-1]
                label_files.append(filename)
        return label_files

    def read_logs(self):
        if not os.path.exists(self.log_path):
            return []

        log_video_files = []
        with open(self.log_path, 'r') as log_file:
            for line in log_file.readlines():
                log_video_files.append(str(line).split(",")[0])
        return log_video_files

    def save_image(self, frame, video_file_path, index):
        frame = cv2. cvtColor(frame, cv2.COLOR_RGB2BGR)
        base_path = os.path.dirname(video_file_path)

        video_name_long = os.path.basename(video_file_path)
        video_name = video_name_long.split(".")[0]

        folder_path = os.path.join(base_path, "Images")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        file_name = video_name + "_" + str(index) + ".png"
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)
        cv2.imwrite(file_path, frame)

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
    vc = VideoConverter()
    vc.convert_videos()