import os
import boto3

from io_utils import save_video_overview

cloud_access_key         = "minio"
cloud_secret_key         = "miniostorage"
video_bucket_name        = "test-dih4cps"
cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"

video_formats            = ["avi", "mp4"]

## create file structure 
BASEPATH = os.path.dirname(__file__)
videos_path = os.path.join(BASEPATH, "videos")

if not os.path.exists(videos_path):
    os.mkdir(videos_path)

def sort_videos_by_date(videos_list):
    """
    Input:
        list of filenames/filepaths like:
            .../DESCRIPTION_TIMESTAMP.avi
            with TIMESTAMP like year-month-day_hour-minute-second

    Output:
        dict with sorted filenames/filepaths like:
            {
            TIMESTAMP:  [file1, file2, ...]
            ... 
            }
    """

    videos_dict = {}

    for vid in videos_list:
        vid_name = os.path.basename(vid).split(".")[0]
        timestamp_date = vid_name.split("_")[-2]

        if list(videos_dict.keys()).count(timestamp_date) <= 0:
            videos_dict[timestamp_date] = []
        videos_dict[timestamp_date].append(vid)

    return videos_dict

def get_videos_from_cloud(bucket_name=video_bucket_name):
    video_list = []
                    
    s3_resource = boto3.resource('s3',
                    aws_access_key_id = cloud_access_key,
                    aws_secret_access_key = cloud_secret_key,
                    endpoint_url=cloud_url, 
                    verify=False,
                    config=boto3.session.Config(signature_version='s3v4'))
    s3_bucket = s3_resource.Bucket(bucket_name)

    for bucket_object in s3_bucket.objects.all():
        file_key = str(bucket_object.key)
        file_format = file_key.split(".")[-1]
        if video_formats.count(file_format) > 0:
            video_list.append(file_key)
        
    return video_list

video_list = get_videos_from_cloud()
video_dict = sort_videos_by_date(video_list)

for date in video_dict.keys():
    date_dir = os.path.join(videos_path, date)
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)
    save_video_overview(os.path.join(date_dir, "video_overview.txt"), video_dict[date], [], overwrite=True)