def save_video_overview(txt_path, videos_receivable, videos_done, overwrite=True):
    import os
    if os.path.exists(txt_path):
        if overwrite:
            os.remove(txt_path)
        else:
            print("File", txt_path, "allready exists. Use overwrite=True to overwrite file.")
            return 

    with open(txt_path, 'w') as f:
        f.write("Videos_receivable:")
        for v in videos_receivable:
            f.write("\n" + v)
        f.write("\nVideos_done:")
        for v in videos_done:
            f.write("\n" + v)

def load_video_overview(txt_path):
    import os
    if not os.path.exists(txt_path):
        print("File", txt_path, "does not exists.")
        return 

    videos_receivable = []
    videos_done = []
    input_rec = False
    input_done = False

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.count("Videos_receivable:") > 0:
            input_rec = True
            input_done = False
            continue
        if line.count("Videos_done:") > 0:
            input_rec = False
            input_done = True
            continue

        if len(line) > 2 and input_rec:
            videos_receivable.append(str(line.split("\n")[0]))
        if len(line) > 2 and input_done:
            videos_done.append(str(line.split("\n")[0]))
        
    return videos_receivable, videos_done