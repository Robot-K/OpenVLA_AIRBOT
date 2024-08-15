import h5py
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, help='Task name')
parser.add_argument('--camera_name', type=str, help='Camera name')
parser.add_argument('--num_episode', type=int, help='Number of episodes')
args = parser.parse_args()

task_name = args.task_name
cam_name = args.camera_name
num = args.num_episode
path = Path(f'./demonstrations/hdf5') / task_name
path_ = Path(f'./demonstrations') / task_name

print("Recovering video...")
for id in tqdm(range(num)):
    with h5py.File(path / f"episode_{id}.hdf5", "r") as root:
        images = root[f"/observations/images/{cam_name}"][()]
        # JPEG decompression
        images_num = len(images)
        # print(images_num)
        out = cv2.VideoWriter(str(path_ / f"{id}" / f"{cam_name}.avi"), cv2.VideoWriter_fourcc(*'XVID'), 50, (640, 480))
        for index, image in enumerate(images):
            image = cv2.imdecode(image, 1)
            out.write(image)
        out.release()