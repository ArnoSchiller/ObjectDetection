import os
import glob
import shutil

from io_utils import load_video_overview, save_video_overview

model_name = "yolov5l_dsv1"
BASEPATH = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join("..", "trained_models", model_name, "weights", "best.pt")

dirs = glob.glob(os.path.join(BASEPATH, "videos", "*"))
dirs = [os.path.basename(d) for d in dirs if d.count(".") < 1]

result_dir = os.path.join(BASEPATH, "results_" + model_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
dirs = [dirs[0]]
for idx, date_dir in enumerate(dirs):
    os.chdir(result_dir)
    dir_path = os.path.join(result_dir, date_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    txt_path = os.path.join(dir_path, "video_overview.txt")
    if not os.path.exists(txt_path):
        shutil.copy(os.path.join(BASEPATH, "videos", date_dir, "video_overview.txt"), txt_path)
    files_to_process, files_done = load_video_overview(txt_path)

    print(f"Running model for {date_dir} ({idx+1} of {len(dirs)}) ...")
    os.chdir(os.path.join(BASEPATH, "..", "yolov5"))
    for f_idx, f in enumerate(files_to_process):
        fname = os.path.basename(f)
        name = fname.split(".")[0]
        out_dir =  os.path.join("..", "model_usage", "results_" + model_name,date_dir)
        f_path = os.path.join("..", "model_usage", "videos", date_dir, fname)
        print(f"Processing video {f} ({f_idx+1} of {len(files_to_process)})")
        command = f"python detect.py --weights {WEIGHTS_PATH} --source {f_path} --img-size 640 --conf-thres 0.5 --iou-thres 0.5 --save-txt --save-conf --project {out_dir} --name {name}"
        print(command)
        os.system(command)

        files_to_process.remove(f)
        files_done.append(f)
        save_video_overview(txt_path, files_to_process, files_done, overwrite=True)

    print("-"*40)
