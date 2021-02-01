import boto3
import glob
import os

from io_utils import load_video_overview

cloud_access_key      = "minio"
cloud_secret_key      = "miniostorage"
cloud_url             = "https://minio.dih4cps.swms-cloud.com:9000/"
cloud_bucket          = "test-dih4cps"

dirs = glob.glob(os.path.join(os.path.dirname(__file__), "videos", "*"))
dirs = [os.path.basename(d) for d in dirs if d.count(".")<1]
print(dirs)
date_dirs = dirs # ["2020-09-30"]


def download_files(bucket, file_list, save_path):
    s3_client = boto3.client('s3',
                aws_access_key_id = cloud_access_key,
                aws_secret_access_key = cloud_secret_key,
                endpoint_url = cloud_url, 
                verify=False,
                config=boto3.session.Config(signature_version='s3v4'))
    
    for idx, f in enumerate(file_list):
        print(f"Downloading file {idx+1} of {len(file_list)}.", end="\r")
        fname = os.path.basename(f)
        local_path = os.path.join(save_path, fname)
        if not os.path.exists(local_path):
            s3_client.download_file(bucket, f, local_path)

    print("")
    print("Checking downloads..")
    files_in_dir = glob.glob(os.path.join(save_path, "*"))
    files_in_dir = [os.path.basename(f) for f in files_in_dir]
    missing_files = []
    for f in file_list:
        fname = os.path.basename(f)
        if not files_in_dir.count(fname) > 0:
            missing_files.append(f)

    print(f"Downloaded {len(file_list) - len(missing_files)} of {len(file_list)} files ({len(missing_files)} missing)")

    return len(file_list) - len(missing_files)

BASEPATH = os.path.dirname(__file__)

num_sum = 0
for date_dir in date_dirs:
    print(f"Download files from {date_dir}.")
    dir_path = os.path.join(BASEPATH, "videos", date_dir)
    files_to_download, files_done = load_video_overview(os.path.join(dir_path, "video_overview.txt"))
    num_sum += download_files(cloud_bucket, files_to_download, dir_path)

print(f"{num_sum} files downloaded.")