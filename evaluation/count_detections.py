import os, glob

PATH = os.path.abspath("./Object-Detection-Metrics/yolov5l_dsv1_pred")
files = glob.glob(os.path.join(PATH, "*.txt"))

print("Found", len(files), "files")

num_dets = 0
empty = 0
for p in files:
    with open(p, 'r') as f:
        dets = f.readlines()
        empty_file = True 
        for line in dets:
            if line.count("shrimp") > 0:
                num_dets += len(dets)
                empty_file = False
            else:
                if empty_file:
                    empty += 1
        f.close()
print("... with", num_dets, "detections")
print("... and", empty, "empty files")