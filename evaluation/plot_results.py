import matplotlib.pyplot as plt
import os
BASE_PATH = os.path.join(os.path.dirname(__file__), "Object-Detection-Metrics")

results_name = []
results_paths = []
colours = []

"""
results_name.append("yolov5l_dsv1")
results_paths.append(os.path.join(BASE_PATH, "results_yolov5l_dsv1", "results.txt"))
colours.append('-b')

results_name.append("yolov5l_dsv1_gray")
results_paths.append(os.path.join(BASE_PATH, "results_yolov5l_dsv1_gray", "results.txt"))
colours.append('-g')

fig = plt.figure()
ax=fig.add_subplot(111)
ax.grid()

for idx, res_txt in enumerate(results_paths): 

    predictions = []
    recalls = []
    mAP = 0

    with open(res_txt, 'r') as f:
        results = f.readlines()
    for res in results:
        if res.count("Precision: ") > 0:
            res = res.split("[")[1].split("]")[0]
            for pred in res.split(","):
                pred = pred.split("'")[1]
                predictions.append(float(pred))
        if res.count("Recall: ") > 0:
            res = res.split("[")[1].split("]")[0]
            for recall in res.split(","):
                recall = recall.split("'")[1]
                recalls.append(float(recall))
        if res.count("mAP: ") > 0:
            mAP = res.split(" ")[1].split("%")[0]
            
        
    ax.plot(recalls, predictions, colours[idx], label=results_name[idx] + f" (mAP: {mAP}%)")

plt.ylabel("Precision")
plt.xlabel("Recall")
plt.legend()
plt.savefig(f"res_{results_name[0]}_{results_name[1]}.png")
plt.show()
"""

colours = ["-b", "-r", "-g", "--b", "--r", "--g", "ob", "or", "og", "+b", "+r", "+g"]
results_name = "yolov5l_dsv2"

res_txt = os.path.join(BASE_PATH, "results_" + results_name, "results.txt")

fig = plt.figure()
ax=fig.add_subplot(111)
ax.grid()


all_preds = []
all_recs = []
classes = []
aps = []

predictions = []
recalls = []
mAP = 0


with open(res_txt, 'r') as f:
    results = f.readlines()
for res in results:
    if res.count("Precision: ") > 0:
        res = res.split("[")[1].split("]")[0]
        if len(res.split(",")) < 2:
            all_preds.append([])
            continue
        for pred in res.split(","):
            pred = pred.split("'")[1]
            predictions.append(float(pred))
        all_preds.append(predictions)
        predictions = []
    if res.count("Recall: ") > 0:
        res = res.split("[")[1].split("]")[0]
        if len(res.split(",")) < 2:
            all_recs.append([])
            continue
        for recall in res.split(","):
            recall = recall.split("'")[1]
            recalls.append(float(recall))
        all_recs.append(recalls)
        recalls = []
    if res.count("mAP: ") > 0:
        mAP = res.split(" ")[1].split("%")[0]
    if res.count("AP: ") > 0:
        aps.append(res.split(" ")[1].split("%")[0])
    if res.count("Class: ") > 0:
        classes.append(res.split("Class: ")[1].split("\n")[0])

for idx in range(len(classes)):
    ax.plot(all_recs[idx], all_preds[idx], label=classes[idx] + f" (AP: {aps[idx]}%)")

plt.title(f"mAP: {mAP}%")

plt.ylabel("Precision")
plt.xlabel("Recall")
plt.legend()
plt.savefig(f"res_{results_name}.png")
plt.show()