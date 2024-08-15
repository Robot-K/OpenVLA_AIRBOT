# to rearrange the camera images into the desired order

from pathlib import Path
import argparse
import os
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Rearrange camera names. Non-existing camera video will be filled in black.')
parser.add_argument('--task_name', type=str, help='Task name')
parser.add_argument('--before', help='List of camera names before rearrangement', nargs='+')
parser.add_argument('--after', help='List of camera names after rearrangement', nargs='+')
args = parser.parse_args()

task_name = args.task_name

before = args.before
after = args.after

path = Path(f'./demonstrations') / task_name
video = cv2.VideoCapture(str(path / "0" / "0.avi"))
total_frame_cnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)
print(f"Get total frame count={total_frame_cnt}")
for entry in tqdm(os.scandir(path)):
    if entry.is_dir():
        path_ = path / entry.name
        for camera in before:
            if not os.path.exists(path_ / f'{camera}.avi'):
                try:
                    os.system(f'ffmpeg -loglevel error -f lavfi -i color=c=black:s=640x480:r={fps}\
                        -vframes {total_frame_cnt} {path_ / f"{camera}.avi"}')
                except:
                    os.system("exit()")
            os.rename(path_ / f'{camera}.avi', path_ / f'{camera}_.avi')
        for camera in before:
            camera_after = after[before.index(camera)]
            os.rename(path_ / f'{camera}_.avi', path_ / f'{camera_after}.avi') 
