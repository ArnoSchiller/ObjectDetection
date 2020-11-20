Add models to the models/ folder to train new models. 
Necessary files and folders:
    - data 
        - test.record
        - train.record
        - object-detection.pbtxt
    - images 
        - image files to train ob
    - training 
        - model.config
    - pretrained model (e.g. ssd_mobilenet_v1_coco_2018_01_28)

Run the training:
    python train.py --train_dir=training/ --model_name={*model_name*} --logtostderr

    model_name = Name des models (e.g. tf_API_data2_v01)