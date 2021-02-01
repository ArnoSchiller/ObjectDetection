## datasets

This folder contains datasets used for training and python code to handle dataset files. For more informations check the readme.md included in the *python_modules* directory.  
To add new datasets please use the following structure:


*DATASET-NAME*
|-- *VERSION-NAME*
|   |-- voc
|   |   |-- train
|   |   |-- valid
|   |   '-- test
|   |
|   |-- pytorch
|   |   |-- train
|   |   |   |-- images
|   |   |   '-- labels
|   |   |-- valid
|   |   |   |-- images
|   |   |   '-- labels
|   |   |-- test
|   |   |   |-- images
|   |   |   '-- labels
|   |   '-- data.yml
|   |
|   '-- tfrecord
|       |-- train.csv
|       |-- train.tfrecord
|       |-- valid.csv
|       |-- valid.tfrecord
|       |-- test.csv
|       '-- test.tfrecord
|
|-- *VERSION-NAME*
    ...