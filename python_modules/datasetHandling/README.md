These modules are used to handle the minio cloud storage.
The modules have following functionality:

dataset_statustics      Shows a table with the number of labeled data for each dataset

cloud_rename_files      can be used to change the location of a stored file 

download_files          downloads the dataset, optionally with test-train-split

convert_data2_to_data1  copy and adapt the label file of data from v2 to v1

dataset_handler         creates a dataset version with a list of included data files, a csv and a tfrecord file for this version
                        returns informations about the dataset
                        downloads not labeled images or label data without image

data_transformation     convert xml files to one csv file and generate a tfrecord file depending on a csv file 