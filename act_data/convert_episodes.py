import cv2
import numpy as np
import time
import json
import os
import multiprocessing
from threading import Thread
from typing import Union, List

from hdf5er import save_one_episode
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(quaternion):
    r = R.from_quat(quaternion)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return [roll, pitch, yaw]

class Raw2Hdf5(object):
    def __init__(self, root_dir, task_name, json_name, camera_names, episode_num):
        """
        default video type is "avi"; if you want to change it, you can use "set_video_type" method
        """
        self.root_dir = root_dir
        self.json_name = json_name + ".json"
        self.camera_names = camera_names
        if not isinstance(episode_num, int):
            self.start_episode = episode_num[0]
            self.end_episode = episode_num[1]
            self.episode_num = self.end_episode - self.start_episode + 1
        else:
            self.episode_num = episode_num
            self.start_episode = 0
            self.end_episode = episode_num - 1
        if root_dir[-1] != "/":
            root_dir += "/"
        if task_name != "":
            self.task_path = root_dir + task_name + "/"
        else:
            self.task_path = root_dir
        # optional record
        self.other_record = {
            "vel": False,
            "eff": False,
            "base": False,
            "eef_pose": False,
        }
        # output (hdf5) names
        self.name_obs_joint_positions_o = "/observations/qpos"
        self.name_act_joint_positions_o = "/action"
        self.name_obs_eef_pose_o = "/observations/eef_pose"
        self.name_act_eef_pose_o = "/actions/eef_pose"
        self.name_obs_images_o = "/observations/images"
        # input (json and avi) names
        self.name_obs_joint_positions_i = ["/observations/pos_f"]
        self.name_act_joint_positions_i = ["/observations/pos_t"]
        self.name_obs_end_positions_i = ["/observations/endpos_f"]
        self.name_act_end_positions_i = ["/observations/endpos_t"]
        self.name_obs_eef_pose_i = ["/observations/eef_pose_f"]
        self.name_act_eef_pose_i = ["/observations/eef_pose_t"]
        self.name_obs_images_i = "/observations/images"
        # default size
        self.image_shape = (480, 640, 3)
        self.arm_joint_num = 6
        self.eef_joint_num = 1
        self.joint_num = 7
        self.agent_num = 1
        self.states_num = self.joint_num * self.agent_num

    def set_image_shape(self, w, h, grey=False):
        self.image_shape = (h, w, 3) if not grey else (h, w)

    def set_in_names(
        self,
        obs_joint_positions: list,
        act_joint_positions: list,
        obs_end_positions: list,
        act_end_positions: list,
        obs_eef_pose: list,
        act_eef_pose: list,
        obs_images: str,
    ):
        self.name_obs_joint_positions_i = obs_joint_positions
        self.name_act_joint_positions_i = act_joint_positions
        self.name_obs_end_positions_i = obs_end_positions
        self.name_act_end_positions_i = act_end_positions
        self.name_obs_images_i = obs_images
        self.name_obs_eef_pose_i = obs_eef_pose
        self.name_act_eef_pose_i = act_eef_pose
        self.agent_num = len(obs_joint_positions)
        self.states_num = self.joint_num * self.agent_num

    def set_out_names(self, obs_joint_positions, act_joint_positions, obs_images):
        # TODO: now not support for modifying
        self.name_obs_joint_positions_o = obs_joint_positions
        self.name_act_joint_positions_o = act_joint_positions
        self.name_obs_images_o = obs_images

    def set_joints_num(self, arm, eef):
        """Set the number of joints of the robot arm and end effector (single robot)"""
        self.arm_joint_num = arm
        self.eef_joint_num = eef
        self.joint_num = arm + eef
        self.states_num = self.joint_num * self.agent_num

    def init(self):
        # the data size should be the same for all episodes
        print("Start to init data")
        # get raw json data and info
        joint_num, self.episode_len, self.keys = self.get_data_info()
        assert (
            joint_num == self.joint_num
        ), f"joint_num: {joint_num} not equal to {self.joint_num} set manually"
        print(f"episode_len: {self.episode_len}")
        # print("keys:", self.keys)
        # init basic data
        self.data = {
            self.name_obs_joint_positions_o: np.zeros(
                (self.episode_len, self.states_num)
            ),
            self.name_act_joint_positions_o: np.zeros(
                (self.episode_len, self.states_num)
            ),
        }
        for name in self.camera_names:
            self.data[self.name_obs_images_o + "/" + name] = [
                np.zeros(self.image_shape)
            ] * self.episode_len
        self.video_type = "avi"
        print("Init data done.")

    def check(self, possible_camera_names, possible_states_num):
        possible_camera_names = set(possible_camera_names)
        names_set = set(
            [
                *self.name_obs_joint_positions_i,
                *self.name_act_joint_positions_i,
                *self.name_obs_end_positions_i,
                *self.name_act_end_positions_i,
            ]
        )
        assert set(self.camera_names).issubset(
            possible_camera_names
        ), f"camera names: {self.camera_names} not subset of {possible_camera_names}"
        assert (
            self.states_num in possible_states_num
        ), f"states_num: {self.states_num} not in {possible_states_num}"
        assert names_set.issubset(
            set(self.keys)
        ), f"names: {names_set} not subset of data keys: {self.keys}"
        # check whether the last episode exists
        if not os.path.exists(self.task_path + f"{self.end_episode}"):
            raise ValueError(
                f"episode_num: {self.episode_num} not exist in {self.task_path}"
            )
        # check whether the episode_num is the last one
        if os.path.exists(self.task_path + f"{self.episode_num}"):
            print("Note: episode_num not the last one")

    def set_other_record(self, names):
        """Set other record, such as velocity, effort, base, eef_pose."""
        for name in names:
            if name == "vel":
                self.data["/observations/qvel"] = [
                    np.zeros(self.joint_num * self.agent_num)
                ] * self.episode_len
                self.other_record["vel"] = True
            elif name == "eff":
                self.data["/observations/effort"] = [
                    np.zeros(self.joint_num * self.agent_num)
                ] * self.episode_len
                self.other_record["eff"] = True
            elif name == "base":
                self.data["/base_action"] = [np.zeros(2)] * self.episode_len
                self.other_record["base"] = True
            elif name == "eef_pose":
                self.data[self.name_obs_eef_pose_o] = np.zeros((self.episode_len, 6 * self.agent_num))
                self.data[self.name_act_eef_pose_o] = np.zeros((self.episode_len, 6 * self.agent_num))
                self.other_record["eef_pose"] = True

    def set_video_type(self, video_type):
        """Set video type, such as "avi" or "mp4"."""
        possible_video_type = ["avi", "mp4"]
        if video_type not in possible_video_type:
            raise ValueError(f"video_type: {video_type} not in {possible_video_type}")
        self.video_type = video_type

    def get_data_info(self) -> tuple:
        """Get the data info by reading the first episode's json file."""
        episode = self.start_episode
        file_path = self.task_path + f"{episode}/" + self.json_name
        with open(file_path) as f_obj:
            raw_json_data = json.load(f_obj)
        eef_data = raw_json_data[self.name_obs_end_positions_i[0]][0]
        if isinstance(eef_data, (float, int)):
            eef_joint_num = 1
        else:
            eef_joint_num = len(eef_data)
        joint_num = (
            len(raw_json_data[self.name_obs_joint_positions_i[0]][0]) + eef_joint_num
        )
        episode_len = len(raw_json_data[self.name_obs_joint_positions_i[0]])
        keys = list(raw_json_data.keys())
        return joint_num, episode_len, keys

    def _read_json_data(self, episode):
        """self.data中 Read the data in the json file and store it in self.data."""
        file_path = self.task_path + f"{episode}/" + self.json_name
        with open(file_path) as f_obj:
            json_data = json.load(f_obj)
        for n in range(self.agent_num):
            # store the basic data
            pos_f = np.array(json_data[self.name_obs_joint_positions_i[n]])
            endpos_f = np.array(json_data[self.name_obs_end_positions_i[n]])
            pos_t = np.array(json_data[self.name_act_joint_positions_i[n]])
            endpos_t = np.array(json_data[self.name_act_end_positions_i[n]])
            slices = slice(n * self.joint_num, (n + 1) * self.joint_num)
            self.data[self.name_obs_joint_positions_o][:, slices] = np.hstack(
                (pos_f, endpos_f.reshape(-1, 1))
            )
            self.data[self.name_act_joint_positions_o][:, slices] = np.hstack(
                (pos_t, endpos_t.reshape(-1, 1))
            )
            # store the extra data
            
            if self.other_record["eef_pose"]:
                for item_obs, item_act in zip(json_data[self.name_obs_eef_pose_i[n]], json_data[self.name_act_eef_pose_i[n]]):
                    item_obs[1] = quaternion_to_euler(item_obs[1])
                    item_act[1] = quaternion_to_euler(item_act[1])
                    
                eef_pose_f = np.array(json_data[self.name_obs_eef_pose_i[n]])
                eef_pose_t = np.array(json_data[self.name_act_eef_pose_i[n]])
                self.data[self.name_obs_eef_pose_o][:, slices] = np.concatenate(
                    [eef_pose_f[:, 0, :], eef_pose_f[:, 1, :]], axis=1
                )
                self.data[self.name_act_eef_pose_o][:, slices] = np.concatenate(
                    [eef_pose_t[:, 0, :], eef_pose_t[:, 1, :]], axis=1
                )
            if self.other_record["vel"]:  # TODO: support for modifying name
                vel_f = np.array(json_data["/observations/vel_f"])
                endvel_f = np.array(json_data["/observations/endvel_f"])
                self.data["/observations/qvel"][:, slices] = np.hstack(
                    (vel_f, endvel_f.reshape(-1, 1))
                )
            if self.other_record["eff"]:  # TODO: support for modifying name
                eff_f = np.array(json_data["/observations/eff_f"])
                endeff_f = np.array(json_data["/observations/endeff_f"])
                self.data["/observations/effort"][:, slices] = np.hstack(
                    (eff_f, endeff_f.reshape(-1, 1))
                )
            if self.other_record["base"]:
                self.data["/base_action"] = json_data["/base_action"]

        assert self.data[self.name_obs_joint_positions_o].shape == (
            self.episode_len,
            self.states_num,
        ), f"states shape: {self.data[self.name_obs_joint_positions_o].shape} not equal to ({self.episode_len}, {self.states_num})"

    def _read_video_images(self, episode):
        """Read the images in the video file and store them in self.data."""
        for camera_name in self.camera_names:
            video_path = (
                self.task_path + f"{episode}/" + camera_name + f".{self.video_type}"
            )
            # print(time.time())
            cap = cv2.VideoCapture(video_path)
            # print(time.time())
            # frame_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print("all frame:", frame_all)
            # image_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
            #     cv2.CAP_PROP_FRAME_HEIGHT
            # )
            # assert (
            #     frame_all == self.episode_len
            # ), f"frame count: {frame_all} not equal to episode lenth {self.episode_len}"
            # assert image_size == (
            #     640,
            #     480,
            # ), f"image size: {image_size} not equal to (640, 480)"
            if not cap.isOpened():
                raise Exception("Video not opened.")
            frame_count = 0
            while cap.isOpened():
                # start_time = time.time()
                ret, frame = cap.read()
                # end_time = time.time()
                # print("read one image time", end_time - start_time)
                if ret:
                    self.data[self.name_obs_images_o + f"/{camera_name}"][
                        frame_count
                    ] = frame
                    frame_count += 1
                    now_time = time.time()
                    # print(f"process one image time {now_time - end_time}")
                else:
                    # assert (
                    #     frame_count == frame_all
                    # ), "frame count not equal to frame all"
                    break
            # print(time.time())
            cap.release()
            # print(time.time())

    def _save_one(self, episode, dir=None):
        start_time = time.time()
        self._read_json_data(episode)
        print(f"read json {time.time() - start_time}")
        self._read_video_images(episode)
        if dir is None:
            dir = self.task_path
        mid_time = time.time()
        save_one_episode(
            self.data,
            self.camera_names,
            dir,
            f"episode_{episode}",
            overwrite=True,
            no_base=not self.other_record["base"],
            no_effort=not self.other_record["eff"],
            no_velocity=not self.other_record["vel"],
            no_pose=not self.other_record["eef_pose"],
            compress=True if self.video_type == "avi" else False,
            states_num=self.states_num,
        )
        end_time = time.time()
        print(
            f"episode_{episode}: read data {mid_time - start_time} s and save data {end_time - mid_time} s"
        )

    def save(self, dir=None, multi_threads=False):
        """Save data to hdf5 for each episode."""
        if multi_threads:
            threads_list: List[Thread] = []
            for episode in range(self.start_episode, self.end_episode + 1):
                thread = Thread(target=self._save_one, args=(episode, dir))
                threads_list.append(thread)
                thread.start()
            for t in threads_list:
                t.join()
        else:
            for episode in range(self.start_episode, self.end_episode + 1):
                self._save_one(episode, dir)


def convert_episode(args: dict, episode_id: Union[int, tuple], multi_threads=None):
    # data path
    root_dir = args.root_dir
    task_name = args.task_name
    json_name = args.json_name
    video_type = args.video_type
    hdf5_dir = args.save_dir
    # convert camera names to list
    camera_names = args.camera_names.split(",")
    robot_num = args.robot_num
    if isinstance(episode_id, int):
        episodes = (episode_id, episode_id)
    else:  # e.g. (1, 4)
        episodes = episode_id
    datar = Raw2Hdf5(root_dir, task_name, json_name, camera_names, episodes)
    if robot_num == 1:
        datar.set_in_names(
            ["/observations/pos_f"],
            ["/observations/pos_t"],
            ["/observations/endpos_f"],
            ["/observations/endpos_t"],
            ["/observations/eef_pose_f"],
            ["/observations/eef_pose_t"],
            ["/observations/images"],
        )
    elif robot_num == 2:
        datar.set_in_names(
            ["/observations/pos_f_left", "/observations/pos_f_right"],
            ["/observations/pos_t_left", "/observations/pos_t_right"],
            ["/observations/endpos_f_left", "/observations/endpos_f_right"],
            ["/observations/endpos_t_left", "/observations/endpos_t_right"],
            ["/observations/eef_pose_f_left", "/observations/eef_pose_f_right"],
            ["/observations/eef_pose_t_left", "/observations/eef_pose_t_right"],
            ["/observations/images"],
        )
    else:
        raise NotImplementedError(f"robot_num: {robot_num} not implemented")
    datar.set_image_shape(640, 480, grey=False)
    datar.init()
    datar.set_video_type(video_type)
    datar.check(
        [
            "cam_high",
            "cam_low",
            "cam_left_wrist",
            "cam_right_wrist",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ],
        [7, 14],
    )
    if args.other_record != "":
        datar.set_other_record(args.other_record.split(","))
    if not args.not_add_task_name:
        if hdf5_dir[-1] not in ["/", ""]:
            hdf5_dir += "/"
        hdf5_dir += task_name + "/"
    try:
        datar.save(hdf5_dir)
    except ValueError as e:
        if multi_threads is not None:
            multi_threads["ok"] = False
        else:
            raise Exception(f"{e}")


def convert_episode_multi(args: dict, episodes: tuple):
    threads_list: List[Thread] = []
    assert len(episodes) == 2
    multi_threads_flag = [{"ok": True}] * (episodes[1] - episodes[0] + 1)
    all_episodes = list(range(episodes[0], episodes[1] + 1))
    for index, episode in enumerate(all_episodes):
        thread = Thread(
            target=convert_episode, args=(args, episode, multi_threads_flag[index])
        )
        threads_list.append(thread)
        thread.start()
    for index, t in enumerate(threads_list):
        t.join()
        all_try = args.try_times
        while not multi_threads_flag[index]["ok"] and all_try > 0:
            print("尝试重试")
            td = Thread(
                target=convert_episode,
                args=(args, all_episodes[index], multi_threads_flag[index]),
            )
            td.join()
            all_try -= 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("convert cpp data to python data")
    parser.add_argument(
        "-rd",
        "--root_dir",
        type=str,
        default=f"{os.getcwd()}/demonstrations/raw/",
        help="root dir of the raw data",
    )
    parser.add_argument(
        "-tn", "--task_name", type=str, default="test_task", help="task name"
    )
    parser.add_argument(
        "-jn",
        "--json_name",
        type=str,
        default="records",
        help="json file name of the data",
    )
    parser.add_argument(
        "-cn", "--camera_names", type=str, default="0,1,2", help="camera names"
    )
    parser.add_argument(
        "-se",
        "--start_episode",
        type=int,
        default=0,
        help="the id of the starting episode, inclusive",
    )
    parser.add_argument(
        "-ee",
        "--end_episode",
        type=int,
        default=1,
        help="the id of the ending episode, inclusive",
    )
    parser.add_argument(
        "-vt",
        "--video_type",
        type=str,
        default="avi",
        help="video type of the raw images data",
    )
    parser.add_argument(
        "-mp",
        "--multi_processing",
        type=int,
        default=10,
        help="use multi processing to convert data",
    )
    parser.add_argument(
        "-mt",
        "--multi_threads",
        type=int,
        default=5,
        help="use multi threads to convert data in each processing",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        type=str,
        default=f"{os.getcwd()}/demonstrations/hdf5/",
        help="save dir of the target hdf5 data",
    )
    parser.add_argument(
        "-or",
        "--other_record",
        type=str,
        default="",
        help="other info to be converted, e.g. vel,eff,base",
    )
    parser.add_argument(
        "-nat",
        "--not_add_task_name",
        action="store_true",
        help="not auto add task name for more custom save dir",
    )
    parser.add_argument(
        "-rn",
        "--robot_num",
        type=int,
        default=1,
        help="number of robots, default is 1",
    )
    parser.add_argument(
        "-tt",
        "--try_times",
        type=int,
        default=2,
        help="try times after converting failed",
    )
    args, unknown = parser.parse_known_args()
    episodes = (args.start_episode, args.end_episode)
    episodes_range = list(range(args.start_episode, args.end_episode + 1))
    multi_processing = args.multi_processing
    """
    Calculate how many threads each process has
    If the number of tasks does not exceed the maximum number of processes, then each process thread is 1
    If the number of tasks exceeds the maximum number of processes, the number of threads per process is the number of tasks divided by the maximum number of processes.
    If there is a remainder after dividing the number of tasks by the maximum number of processes, the number of threads for each process is increased by 1.
    For example: the number of tasks is 10 and the maximum number of processes is 4, then the number of threads per process is 3,3,3,1
    If the calculated number of threads per task exceeds the maximum number of threads, the number of threads per process is the maximum number of threads.
    For example: the number of tasks is 10, the maximum number of processes is 2, and the maximum number of threads is 4, then the number of threads per process is 4, 4, 2
    """
    chunk_size = len(episodes_range) // multi_processing
    chunk_size = (
        chunk_size + 1 if len(episodes_range) % multi_processing != 0 else chunk_size
    )
    chunk_size = args.multi_threads if chunk_size > args.multi_threads else chunk_size
    episodes_chunked = []
    for i in range(0, len(episodes_range), chunk_size):
        slices = slice(i, i + chunk_size)
        episodes_chunked.append((episodes_range[slices][0], episodes_range[slices][-1]))
    start_time = time.time()
    if multi_processing <= 1:
        convert_episode_multi(args, episodes)
    else:
        pool = multiprocessing.Pool(multi_processing)
        pool.starmap(  # Blocking call
            convert_episode_multi, [(args, episodes) for episodes in episodes_chunked]
        )
        pool.close()
        pool.join()  # wait for all processes to finish
    print(f"All done in {time.time() - start_time} seconds.")
