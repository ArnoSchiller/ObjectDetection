import os, sys
sys.path.append(os.path.abspath("../python_modules/custom_models"))


DATASET_NAME = "dataset-v2-dih4cps"
DATASET_VERSION = "version_1"
DATASET_PATH = os.path.abspath(f"../datasets/{DATASET_NAME}/{DATASET_VERSION}/")
CSV_PATH = os.path.join(DATASET_PATH, "test.csv")
IMG_PATH = os.path.join(DATASET_PATH, "voc", "test")

tf_model = False
if tf_model:
    MODEL_NAME = "efficientdet0_dsv1"
    MODEL_PATH = os.path.abspath(f"../trained_models/{MODEL_NAME}/")
    MODEL_WEIGHTS = MODEL_NAME
    from tf_api_model import detect
    
else:
    MODEL_NAME = "yolov5l_dsv2_distance"
    MODEL_PATH = os.path.abspath(f"../trained_models/{MODEL_NAME}/")
    MODEL_WEIGHTS = os.path.join(MODEL_PATH, "weights", "best.pt")
    from pt_model import detect
    
GT_PATH = os.path.join(".", "Object-Detection-Metrics", MODEL_NAME + "_gt")
GT_REL_PATH = os.path.join(".", "Object-Detection-Metrics", MODEL_NAME + "_gt_rel")
PRED_PATH = os.path.join(".", "Object-Detection-Metrics", MODEL_NAME + "_pred")
PRED_REL_PATH = os.path.join(".", "Object-Detection-Metrics", MODEL_NAME + "_pred")
OUT_PATH = os.path.join(".", "results_" + MODEL_NAME)

### CREATE GROUNDTRUTH OUTPUTS 
if not (os.path.exists(GT_PATH) or os.path.exists(GT_REL_PATH)):
    print("Creating dir including groundtruths.")

    ## load groundtruths from csv
    import csv
    gts = {}
    print(f"Loading groundtruths from csv file ({CSV_PATH}).")
    with open(CSV_PATH) as csv_file:
        csv_reader_object = csv.DictReader(csv_file)
        for row in csv_reader_object:
            file_name = row['filename'].split(".")[0]
            if not list(gts.keys()).count(file_name):
                gts[file_name] = []
            gt_dict = {}
            gt_dict['class'] = row['class']
            gt_dict['xmin'] = row['xmin']
            gt_dict['ymin'] = row['ymin']
            gt_dict['xmax'] = row['xmax']
            gt_dict['ymax'] = row['ymax']
            gt_dict['width'] = row['width']
            gt_dict['height'] = row['height']
            gts[file_name].append(gt_dict)
        
        ## create gt outputs
        # crate dirs for groundtruths
        os.mkdir(GT_PATH)
        #os.mkdir(GT_REL_PATH)

        # write a txt for every file including the gts like 
        # {class} {xmin} {ymin} {xmax} {ymax}

        num_files = len(list(gts.keys()))
        for idx, file_name in enumerate(gts.keys()):
            if idx != num_files:
                print(f"    Copying file {idx+1} of {len(list(gts.keys()))}", end='\r')
            else:
                print(f"    Copying file {idx+1} of {len(list(gts.keys()))}")

            with open(os.path.join(GT_PATH, file_name + ".txt"), "w") as f:
                for gt_box in gts[file_name]:
                    xmin, ymin, xmax, ymax = gt_box['xmin'], gt_box['ymin'], gt_box['xmax'], gt_box['ymax']
                    f.write(f"{gt_box['class']} {xmin} {ymin} {xmax} {ymax}\n")
            """
            with open(os.path.join(GT_REL_PATH, file_name + ".txt"), "w") as f:
                for gt_box in gts[file_name]:
                    xmin = float(gt_box['xmin']) / float(gt_box['width']) 
                    ymin = float(gt_box['ymin']) / float(gt_box['height'])
                    xmax = float(gt_box['xmax']) / float(gt_box['width'])
                    ymax = float(gt_box['ymax']) / float(gt_box['height'])
                    f.write(f"{gt_box['class']} {xmin} {ymin} {xmax} {ymax}\n")
            """
else:
    print("Groundtruth for", MODEL_NAME, "exists.")

## CREATE PREDICTION OUTPUTS
if not (os.path.exists(PRED_PATH) or os.path.exists(PRED_REL_PATH)):
    print("Creating dir including groundtruths.")

    ## load filenames from csv
    import csv
    gts = {}
    print(f"    Loading filenames from csv file ({CSV_PATH}).")
    with open(CSV_PATH) as csv_file:
        csv_reader_object = csv.DictReader(csv_file)
        for row in csv_reader_object:
            file_name = row['filename'].split(".")[0]
            if not list(gts.keys()).count(file_name):
                gts[file_name] = []
    

        ## create pred outputs
        # crate dirs for predictions
        #os.mkdir(PRED_PATH)
        IMG_PATH = os.path.join("..", "datasets/dataset-v2-dih4cps/version_1/voc/test")
        MODEL_WEIGHTS = os.path.join("..", "trained_models/yolov5l_dsv2/weights/best.pt")
        print(f"python ..\yolov5\detect.py --source {IMG_PATH} --weights {MODEL_WEIGHTS} --img-size 640 --conf-thres 0.5 --iou-thres 0.5 --save-txt --save-conf")
        # detect(IMG_PATH, MODEL_WEIGHTS, 640, (640,480), txt_out_path=PRED_PATH)
    
else:
    print("Prediction for", MODEL_NAME, "exists.")

OUT_PATH = os.path.join(".", "Object-Detection-Metrics", "results_" + MODEL_NAME)
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

OUT_PATH = os.path.join(".", "results_" + MODEL_NAME)
GT_PATH = os.path.join(".", MODEL_NAME + "_gt")
GT_REL_PATH = os.path.join(".", MODEL_NAME + "_gt_rel")
PRED_PATH = os.path.join(".", MODEL_NAME + "_pred")
PRED_REL_PATH = os.path.join(".", MODEL_NAME + "_pred")
print("-"*80)
print("cd Object-Detection-Metrics")
print(f"python pascalvoc.py -gt {GT_PATH} -det {PRED_PATH} -t  0.0000001 -gtformat xyrb -detformat xywh -gtcoords abs -detcoords rel -imgsize 640,480 -sp {OUT_PATH}")
print("-"*80)