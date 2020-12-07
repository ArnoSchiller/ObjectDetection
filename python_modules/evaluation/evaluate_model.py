import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import csv
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from datasetHandling.dataset_handler import DatasetHandler
from helpers.iou import compute_iou as calc_iou
from config import DATASET_PATH

class ModelEvaluator():
    def __init__(self):
        ## ADD MODEL SELECTION?
        self.dataset_name = "dataset-v1-dih4cps"
        self.dataset_version = "version_2020-12-01"

        self.with_plots = True
        
    def calculate_mAP(self, classes, results):
        res_classes = []
        sum_ap = 0
        for class_name in classes:
            if list(results.keys()).count(class_name) > 0:
                if list(results[class_name].keys()).count("AP") > 0:
                    res_classes.append(class_name)
                    sum_ap += results[class_name]['AP']

        return sum_ap / len(res_classes) 

    def calculate_11_points_AP(self, precisions, recalls):

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        AP_prec = []
        AP_recalls = np.linspace(0.0, 1.0, 11)
        for recall_level in AP_recalls:
            try:
                args= np.argwhere(recalls>recall_level).flatten()
                prec= max(precisions[args])

                """
                print(recalls,"Recall")
                print(recall_level,"Recall Level")
                print(args, "Args")
                print(prec, "precision")
                """
            except ValueError:
                prec=0.0
            AP_prec.append(prec)

            
        if self.with_plots:
            self.plot_precision_recall(precisions, recalls, AP_prec, AP_recalls)
            
        return np.mean(AP_prec) 
    
    def calculate_precision_recall(self, img_results):
        """Calculates precision and recall from the set of images
        Args:
            img_results (dict): dictionary formatted like:
                {
                    'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                    'img_id2': ...
                    ...
                }
        Returns:
            tuple: of floats of (precision, recall)
        """
        true_positive=0
        false_positive=0
        false_negative=0
        for img_id, res in img_results.items():
            true_positive +=res['true_positive']
            false_positive += res['false_positive']
            false_negative += res['false_negative']
        try:
            precision = true_positive/(true_positive + false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
        return (precision, recall)

    def evaluate_model(self, model):
        self.pred_logger = PredictionLogger(model)

        gt_boxes = self.pred_logger.load_evaluation_set(self.dataset_name, self.dataset_version, "csv")

        pred_boxes = {}
        for class_name in gt_boxes.keys():
            pred_boxes[class_name] = {}
            for image_id in gt_boxes[class_name].keys():
                pred_boxes[class_name][image_id] = {}
                boxes = gt_boxes[class_name][image_id]
                scores = np.random.rand(len(boxes)).tolist()
                pred_boxes[class_name][image_id]["boxes"] = boxes
                pred_boxes[class_name][image_id]["scores"] = scores

        results = {}
        classes = gt_boxes.keys()
        for class_name in classes:
            results[class_name] = self.evalute_predictions(gt_boxes[class_name], pred_boxes[class_name])

        results['mAP'] = self.calculate_mAP(classes, results)
        return results

    def evalute_predictions(self, gt_boxes, pred_boxes, iou_thr=0.5):
        """
        Inputs:
            gt_boxes= 
            {
                'img_id1': [[x1, y1, x2, y2], [x1, y1, x2, y2]]
                'img_id2': [[x1, y1, x2, y2]]
                ...
            }
   
            pred_boxes=
            {
                'img_id1': {"boxes":[[x1, y1, x2, y2], [x1, y1, x2, y2]], "scores": [0.1, 0.5]}
                'img_id2': {"boxes":[[x1, y1, x2, y2]], "scores": [0.9]}
                ...
            }

            model_score_thr:  Threshold  to classify a recognition as correct (float)

            iou_thr:    Threshold  to classify a bouding box as correct (float)
        """

        precisions = []
        recalls = []
        model_thrs = []
        total_img_res = {} 

        # Sort the predicted boxes in descending order (lowest scoring boxes first):
        for img_id in pred_boxes.keys():
            arg_sort = np.argsort(pred_boxes[img_id]['scores'])
            pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
            pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

        pred_boxes_pruned = deepcopy(pred_boxes)

        # Get every reached model score and loop over these scores
        model_scores = self.get_model_scores(pred_boxes)
        sorted_model_scores= sorted(model_scores.keys())# Sort the predicted 
        for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
            model_thrs.append(model_score_thr)
            img_ids = gt_boxes.keys() 
            for img_id in img_ids:
                # load ground truth bounding boxes
                gt_boxes_img = gt_boxes[img_id]

                # load predicted bb where the score is higher than the threshold 
                pred_boxes_img = pred_boxes_pruned[img_id]
                box_scores = pred_boxes_img['scores']
                start_idx = 0
                for score in box_scores:
                    if score <= model_score_thr:
                        start_idx += 1
                    else:
                        break 

                # Remove boxes, scores of lower than threshold scores:
                pred_boxes_img['scores'] = pred_boxes_img['scores'][start_idx:]
                pred_boxes_img['boxes'] = pred_boxes_img['boxes'][start_idx:]

                img_res = self.get_single_image_results(gt_boxes_img, pred_boxes_img['boxes'], iou_thr)
                
                img_results = {}
                img_results[img_id] = img_res
                total_img_res[img_id] = img_res

                prec, rec = self.calculate_precision_recall(img_results)
                precisions.append(prec)
                recalls.append(rec)
        
        total_prec, total_rec = self.calculate_precision_recall(total_img_res)

        
        ap = self.calculate_11_points_AP(precisions, recalls)

        evaluation = {}
        evaluation['precisions'] = precisions
        evaluation['recalls'] = recalls
        evaluation['model_thrs'] = model_thrs
        evaluation['AP'] = ap

        return evaluation

    def get_model_scores(self, pred_boxes):
        """Creates a dictionary from model_scores to image ids.
        Args:
            pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
        Returns:
            dict: keys are model_scores and values are image ids (usually filenames)
        """
        model_score={}
        for img_id, val in pred_boxes.items():
            for score in val['scores']:
                if score not in model_score.keys():
                    model_score[score]=[img_id]
                else:
                    model_score[score].append(img_id)
        return model_score

    def get_single_image_results(self, gt_boxes, pred_boxes, iou_thr):
        """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
        Args:
            gt_boxes (list of list of floats): list of locations of ground truth
                objects as [xmin, ymin, xmax, ymax]
            pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`) and 'scores' (list of floats)
            iou_thr (float): value of IoU to consider as threshold for a
                true prediction
        Returns:
            dict: true positives (int), false positives (int), false negatives (int)
        """
        all_pred_indices = range(len(pred_boxes))
        all_gt_indices = range(len(gt_boxes))

        # No objects to detect, every detection is false positive (FP)
        if len(all_gt_indices) == 0:
            tp = 0
            fp = len(all_pred_indices)
            fn = 0
            return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}

        # No object was detected, every object was omitted (false negative - FN)
        if len(all_pred_indices) == 0:
            tp = 0
            fp = 0
            fn = len(all_gt_indices)
            return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}

        gt_idx_thr = []             # indices of gt boxes with a IoU > thresh
        pred_idx_thr = []           # indices of pred boxes with a IoU > thresh
        ious = []                   # IoU values > thresh for every combination

        # Calculate the Intersection over Union for every bounding box combination between gt and pred, if the IoU > thresh, save the indices and the IoU value 
        for ipb, pred_box in enumerate(pred_boxes):
            for igb, gt_box in enumerate(gt_boxes):
                iou = calc_iou(gt_box, pred_box)
                if iou >iou_thr:
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)
        
        # sort the indices by their IoU values 
        iou_sort = np.argsort(ious)[::-1]

        if len(iou_sort) == 0:
            tp = 0
            fp = len(all_pred_indices)
            fn = len(all_gt_indices)
            return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        else:
            # Match bounding boxes, if multiple prediction refer to one object, the box with the higher IoU will be selcted 
            gt_match_idx = []
            pred_match_idx = []

            for idx in iou_sort:
                gt_idx = gt_idx_thr[idx]
                pr_idx = pred_idx_thr[idx]

                # If the boxes are unmatched, add them to matches
                if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                    gt_match_idx.append(gt_idx)
                    pred_match_idx.append(pr_idx)

            # number of matched boxes -> correct detections (TP)
            tp= len(gt_match_idx)
            # number predicted boxes - number matches -> false detections (TP)
            fp= len(pred_boxes) - len(pred_match_idx)   
            # number ground truth boxes -> omitted detection (FN)
            fn = len(gt_boxes) - len(gt_match_idx)
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    
    def plot_precision_recall(self, precisions, recalls, AP_prec, AP_recall):
        precisions = np.array(precisions)
        recalls = np.array(recalls)

        pr_dict = {}
        for index, recall in enumerate(recalls):
            if not list(pr_dict.keys()).count(recall) > 0:
                pr_dict[recall] = []    
            if not pr_dict[recall].count(precisions[index]):
                pr_dict[recall].append(precisions[index])
        
        sorted_recalls = sorted(pr_dict.keys())
        print(sorted_recalls)

        recalls = []
        precisions = []
        for recall in sorted_recalls:
            for precision in pr_dict[recall]:
                recalls.append(recall)
                precisions.append(precision)
        
        plt.plot(recalls, precisions)
        plt.plot(AP_recall, AP_prec, '.')
        plt.xlabel("Recall")
        plt.xlabel("Precision")
        plt.show()


import json
from config import BASE_PATH
class PredictionLogger():
    def __init__(self, model_name):
        log_path = os.path.join(BASE_PATH, "evaluation")
        log_ext = "_log.json"
        self.model_log_path = os.path.join(log_path, model_name + log_ext)
    
        self.dataset_handler = DatasetHandler()

    def load_evaluation_set(self, dataset_name, dataset_version, mode="csv"):

        if mode == "csv":
            file_path = os.path.join(DATASET_PATH, dataset_name, dataset_version, "eval.csv")
            
            labelmap_path, not_downloaded_files = self.dataset_handler.download_evaluation_set(dataset_name, dataset_version, "csv")
            print("Loading evalation set from {} ({})".format(dataset_name, dataset_version)) 

            # load csv file
            csv_path = os.path.join(DATASET_PATH, dataset_name, dataset_version, "eval.csv")

            
            gt_dict = {}
            classes = self.dataset_handler.get_classes_from_labelmap(labelmap_path)
            for class_name in classes:
                gt_dict[class_name] = {}

            with open(csv_path) as csvfile:
                    csv_reader_object = csv.DictReader(csvfile)
                    for row in csv_reader_object:
                        class_name = row['class']
                        file_name = row['filename']
                        bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                        if list(gt_dict[class_name].keys()).count(file_name) > 0:
                            gt_dict[class_name][file_name].append(bbox)
                        else:    
                            gt_dict[class_name][file_name] = [bbox]
            return gt_dict        
            
        elif mode == "xml":
            print("xml - not implemented yet.")
    #####



    def load_predictions(self):
        if not os.path.exists(self.model_log_path):
            print("Can not find model log.")
            return
        
        with open(self.model_log_path, "r") as read_file:
            pred_dict = json.load(read_file)
            print(pred_dict)

        return pred_dict['gt_boxes'], pred_dict['pred_boxes']

    def save_pred_log(self, predictions_dict):
        """
        {
            'gt_boxes' : {  
                'img_id1': [[x1, y1, x2, y2], [x1, y1, x2, y2]]
                'img_id2': [[x1, y1, x2, y2]]
                ...
            }
   
            'pred_boxes' : {
                'img_id1': {"boxes":[[x1, y1, x2, y2], [x1, y1, x2, y2]], "scores": [0.1, 0.5]}
                'img_id2': {"boxes":[[x1, y1, x2, y2]], "scores": [0.9]}
                ...
            }
        }
        """
        #if not os.path.exists(self.model_log_path):
        with open(self.model_log_path, 'w') as f:
            json.dump(predictions_dict, f)

def test_calculation():
    """
    #GT Boxes
    gt_boxes= {"img_00285.png": [[480, 457, 515, 529], [637, 435, 676,
    536]]}
    #Pred Boxes
    pred_boxes={"img_00285.png": {"boxes": [[330, 463, 387, 505], [356,
    456, 391, 521], [420, 433, 451, 498], [328, 465, 403, 540], [480,
    477, 508, 522], [357, 460, 417, 537], [344, 459, 389, 493], [485,
    459, 503, 511], [336, 463, 362, 496], [468, 435, 520, 521], [357,
    458, 382, 485], [649, 479, 670, 531], [484, 455, 514, 519], [641,
    439, 670, 532]], "scores": [0.0739, 0.0843, 0.091, 0.1008, 0.1012,
    0.1058, 0.1243, 0.1266, 0.1342, 0.1618, 0.2452, 0.8505, 0.9113,
    0.972]}}
    """
    """
    #GT Boxes
    gt_boxes= {
        "img_1": [[10, 10, 30, 30]],
        "img_2": [[10, 10, 30, 30]]
        }
    #Pred Boxes
    pred_boxes={
        "img_1": {"boxes": [[10, 10, 30, 30], [20, 20, 40, 40], [20, 20, 40, 40]], "scores": [0.9, 0.9999, 0.9544]},
        "img_2": {"boxes": [[10, 10, 30, 30]], "scores": [0.9]}
        }

    pl = PredictionLogger("model")
    gt, pred = pl.load_predictions()
    """
    me = ModelEvaluator()
    res = me.evaluate_model("")

    for class_name in ['shrimp']:
        print("Class: ", class_name)
        print("    Precisions:   ", res[class_name]['precisions'][1:10], "...")
        print("    Recalls:      ", res[class_name]['recalls'][1:10], "...")
        print("    Thresholds:   ", res[class_name]['model_thrs'][1:10], "...")
        print("    AP:           ", res[class_name]['AP'])
    print("mAP: ", res['mAP'])

    #me.evalute_predictions(gt, pred, model_score_thr=0.6, iou_thr=0.1)



pl = PredictionLogger("model")
p_dict = {
    "gt_boxes" : {  
        "img_id1": [[1, 1, 3, 3], [11, 3, 13, 6]],
        "img_id2": [[6, 2, 10, 5], [2, 5, 4, 7]],
        "img_id3": [[2, 5, 4, 7]],
    },

    "pred_boxes" : {
        "img_id1": {"boxes":[[0, 2, 2, 4], [2, 1, 4, 3], [11, 3, 13, 5], [10, 2, 14, 6]], "scores": [0.9, 0.7, 0.8, 0.6]},
        "img_id2": {"boxes":[[5, 1, 10, 5]], "scores": [0.9]},
        "img_id3": {"boxes":[[5, 1, 10, 5]], "scores": [0.9]},
    }
}

#pl.load_evaluation_set("dataset-v1-dih4cps", "version_2020-12-01")

#pl.save_pred_log(p_dict)
test_calculation()


