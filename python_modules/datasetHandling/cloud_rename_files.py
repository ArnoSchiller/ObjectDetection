import boto3

cloud_access_key         = "minio"
cloud_secret_key         = "miniostorage"
cloud_bucket_name        = "dataset-v1-dih4cps"
cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"
                
s3_resource = boto3.resource('s3',
                aws_access_key_id = cloud_access_key,
                aws_secret_access_key = cloud_secret_key,
                endpoint_url=cloud_url, 
                verify=False,
                config=boto3.session.Config(signature_version='s3v4'))

s3_bucket = s3_resource.Bucket(cloud_bucket_name)

# changing directory of every stored png file to images dir
for bucket_object in s3_bucket.objects.all():
    file_key = str(bucket_object.key)
    if file_key.count("png") > 0:
        new_file_key = "images/" + file_key
        s3_resource.Object(cloud_bucket_name, new_file_key).copy_from(CopySource=cloud_bucket_name + "/" + file_key)
        s3_resource.Object(cloud_bucket_name, file_key).delete()