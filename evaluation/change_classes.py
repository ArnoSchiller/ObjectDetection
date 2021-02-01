import os
import glob 

classes = [
    'shrimp_head_near', 'shrimp_head_middle', 'shrimp_head_far', 'shrimp_tail_near', 'shrimp_tail_middle', 'shrimp_full_far', 'shrimp_full_near', 'shrimp_full_middle', 'shrimp_tail_far', 'shrimp_part_near', 'shrimp_part_middle', 'shrimp_part_far'
]

classes = ["shrimp"]
INPUT_DIR = os.path.join(".", "Object-Detection-Metrics", "yolov5l_dsv1_aug_pred")
OUTPUT_DIR = os.path.join(".", "Object-Detection-Metrics", "yolov5l_dsv1_aug_pred")

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
print("Found", len(files), "files.")

for path in files:
    fname = os.path.basename(path)
    
    with open(path, 'r') as f:
        annots = f.readlines()
        f.close()

    with open(os.path.join(OUTPUT_DIR, fname), 'w') as f:
        for line in annots:
            line = line.split("\n")[0]
            anno = line.split(" ")
            if len(anno) < 5:
                continue
            if str(anno[0]).isnumeric():
                anno[0] = classes[int(anno[0])]
            #else:
            #    anno[0] = anno[0].split("_")[0] + "_" + anno[0].split("_")[-1]

            for a in anno:
                f.write(str(a) + " ")
            f.write("\n")
        f.close()

