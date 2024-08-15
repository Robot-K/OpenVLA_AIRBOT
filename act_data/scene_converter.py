import cv2, argparse, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--filepath")
parser.add_argument("--interval", help="The number between the two sampled frame.", default="100")
parser.add_argument("--hremove", default="0")
parser.add_argument("--config", action='store_true')

args = parser.parse_args()
filepath = args.filepath
video_name = Path(filepath).stem
with_config = args.config
dirpath = Path(filepath).parent / "images" / video_name
os.makedirs(dirpath, exist_ok=True)
vc = cv2.VideoCapture(args.filepath)
interval = eval(args.interval)
hremove = eval(args.hremove)

# determine whether to open normally
if not vc.isOpened():
    print("Cannot open the video.")
    import sys
    sys.exit(-1)

# loop read video frame
cnt = 0
id = 0
while True:
    ret, frame = vc.read()
    if ret == False:
        break
    cnt += 1
    if cnt == interval:
        id += 1
        cnt = 0
        if id < hremove:  # sift out the beginning
            continue
        print(id)
        cv2.imwrite(str(dirpath / f"{id}.jpg"), frame)
        if with_config:
            with open(Path(filepath).parent / "match_config", "a") as f:
                print(f"0/{id}.jpg 1/{id}.jpg", file=f)
                print(f"0/{id}.jpg 2/{id}.jpg", file=f)
                print(f"1/{id}.jpg 2/{id}.jpg", file=f)

vc.release()
