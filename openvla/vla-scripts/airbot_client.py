import requests
import json_numpy
json_numpy.patch()
import json
import sys
import argparse
import numpy as np
import time
from airbot_utils.airbot_play_real_env import make_env, get_image, move_arms, move_grippers
from typing import List
import airbot
from airbot_utils.custom_robot import AssembledRobot
from tqdm import tqdm
import torch
import cv2
from scipy.spatial.transform import Rotation

def get_action(image=np.zeros((224,224,3), dtype=np.uint8), command='do something', unnorm_key='bridge_orig', session=None):
    # unnorm_key should be changed accordingly
    action = session.post(
        "http://localhost:8000/act",
        json={"image": image, "instruction": command, "unnorm_key" : unnorm_key}
    ).json()
    return action

# Function to convert roll, pitch, yaw to quaternion
def euler_to_quaternion(roll, pitch, yaw):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_quat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cmd', "--command", action='store', type=str, help='command to do', default="Do something.", required=False)
    parser.add_argument('-uk', "--unnorm_key", action='store', type=str, help='unnorm key', default="bridge_orig", required=False)
    parser.add_argument('-cw', "--camera_web", action='store', type=str, help='web_camera_indices', default="", required=False)
    parser.add_argument('-cm', "--camera_mv", action='store', type=str, help='machine_vision_camera_indices', default="", required=False)
    parser.add_argument('-can', "--can_buses", action='store', type=str, help='can_bus', default="can0", required=False)
    parser.add_argument('-em', "--eef_mode", action='store', type=str, help='eef_mode', default="gripper")
    parser.add_argument("--time_steps", action='store', type=int, help='max_timesteps', default=20)
    config = vars(parser.parse_args())
    
    camera_names = ['1']
    camera_web = config['camera_web']
    camera_mv = config['camera_mv']
    if camera_mv == "":
        camera_indices = camera_web
    elif camera_web == "":
        camera_indices = camera_mv
    else: camera_indices = camera_mv + ',' + camera_web
    camera_mv_num = len(camera_mv.split(",")) if camera_mv!="" else 0
    camera_web_num = len(camera_web.split(",")) if camera_web!="" else 0
    camera_mask = ["mv"] * camera_mv_num + ["web"] * camera_web_num
    cameras = {name: int(index) for name, index in zip(camera_names, camera_indices.split(','))}
    cameras["mask"] = camera_mask
    
    # init robots
    robot_instances:List[AssembledRobot] = []

    # modify the path to the airbot_play urdf file
    vel = 2.0
    fps = 30
    joint_num = 7
    robot_num = 1
    start_joint = [0] * 7
    can_list = config['can_buses'].split(',')
    eef_mode = config['eef_mode']
    image_mode = 0
    for index, can in enumerate(can_list):
        airbot_player = airbot.create_agent("down", can, vel, eef_mode) # urdf_path
        # airbot_player.set_target_pose([0.3,0,0.2], [0,0,0,1])
        # airbot_player.set_target_end(0)
        # time.sleep(5)
        robot_instances.append(AssembledRobot(airbot_player, 1/fps, 
                                                start_joint[joint_num*index:joint_num*(index+1)]))
    env = make_env(robot_instance=robot_instances, cameras=cameras)
    while True:
        image_list = []  # for visualization
        print('Reset environment...')
        env.reset(sleep_time=1)
        v = input(f'Press Enter to start evaluation or z and Enter to exit...')
        if v == 'z':
            break
        ts = env.reset()
        orientation = [0,0,0]
        try:
            session = requests.session()
            for t in tqdm(range(config['time_steps'])):
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                curr_image = get_image(ts, camera_names, image_mode).cpu().numpy()
                curr_image = np.transpose(np.squeeze(curr_image), (1,2,0))
                height, width = curr_image.shape[:2]
                new_size = min(height, width)
                start_x = (width - new_size) // 2
                start_y = (height - new_size) // 2
                curr_image = curr_image[start_y:start_y + new_size, start_x:start_x + new_size]
                curr_image = cv2.resize(curr_image, (224,224))
                curr_image = (curr_image * 255).astype(np.uint8)
                action = get_action(image = curr_image, command = config['command'], unnorm_key = config['unnorm_key'], session=session)
                scale = 1.0
                current_position, current_orientation = airbot_player.get_current_pose()
                current_end = airbot_player.get_current_end()
                new_position = [
                    current_position[0] + action[0] * scale, # 
                    current_position[1] + action[1] * scale, # 
                    current_position[2] + action[2] * scale, # 
                ]
                orientation[0] += action[3]*scale
                orientation[1] += action[4]*scale
                orientation[2] += action[5]*scale
                new_quaternion = euler_to_quaternion(orientation[0], orientation[1], orientation[2])

                new_end = current_end * 0.5 + action[6] * 0.5
                airbot_player.set_target_pose(new_position, new_quaternion, vel=2.0)
                airbot_player.set_target_end(new_end)
                ts = env.step(action=None)
     
        except KeyboardInterrupt as e:
            print(e)
            print('Evaluation interrupted by user...')