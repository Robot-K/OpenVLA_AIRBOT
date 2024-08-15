import time
import numpy as np
import collections
import dm_env
import torch
from typing import List
from einops import rearrange

from .custom_robot import AssembledRobot, AssembledFakeRobot


class RealEnv:
    """
    Environment for real robot one-manual manipulation
    Action space:      [arm_qpos (6),             # absolute joint position
                        gripper_positions (1),]    # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ arm_qpos (6),          # absolute joint position
                                        gripper_position (1),]  # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ arm_qvel (6),         # absolute joint velocity (rad)
                                        gripper_velocity (1),]  # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"0": (480x640x3),        # h, w, c, dtype='uint8'
                                   "1": (480x640x3),         # h, w, c, dtype='uint8'
                                   "2": (480x640x3),}  # h, w, c, dtype='uint8'
    """

    def __init__(
        self,
        setup_robots=True,
        setup_base=False,
        record_images=True,
        robot_instances: List[AssembledRobot] = None,
        cameras=None,
    ):
        self.airbot_players = robot_instances
        self.robot_num = len(self.airbot_players)
        self.use_base = setup_base
        if setup_robots:
            self.setup_robots()
        if setup_base:
            self.setup_base()
        use_fake_robot = isinstance(self.airbot_players[0], AssembledFakeRobot)

        if record_images:
            if use_fake_robot and not AssembledFakeRobot.real_camera:
                from .robot_utils import ImageRecorderFake as ImageRecorder
            elif isinstance(cameras, dict):
                from .robot_utils import ImageRecorderVideo as ImageRecorder
            else:
                from robot_tools.recorder import ImageRecorderRos as ImageRecorder
                import rospy

                if rospy.get_name() == "/unnamed":
                    rospy.init_node("real_env", anonymous=True)
            self.image_recorder = ImageRecorder(cameras)

        self.reset_position = None
        self.eefs_open = np.array(
            [robot.end_effector_open for robot in self.airbot_players]
        )
        self.eefs_close = np.array(
            [robot.end_effector_close for robot in self.airbot_players]
        )
        if use_fake_robot:
            print("AIRBOT Play Fake Env Created.")
        else:
            print("AIRBOT Play Real Env Created.")

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position

    def setup_robots(self):
        pass

    def setup_base(self):
        pass

    def get_qpos(self, normalize_gripper=False):
        """7 dof: 6 arm joints + 1 gripper joint"""
        qpos = []
        for airbot in self.airbot_players:
            qpos.append(airbot.get_current_joint_positions())
        return np.hstack(qpos)

    def get_qvel(self):
        qvel = []
        for airbot in self.airbot_players:
            qvel.append(airbot.get_current_joint_velocities())
        return np.hstack(qvel)

    def get_effort(self):
        effort = []
        for airbot in self.airbot_players:
            effort.append(airbot.get_current_joint_efforts())
        return np.hstack(effort)

    def get_images(self):
        return self.image_recorder.get_images()

    def get_base_vel(self):
        raise NotImplementedError
        vel, right_vel = 0.1, 0.1
        right_vel = -right_vel  # right wheel is inverted
        base_linear_vel = (vel + right_vel) * self.wheel_r / 2
        base_angular_vel = (right_vel - vel) * self.wheel_r / self.base_r

        return np.array([base_linear_vel, base_angular_vel])

    def get_tracer_vel(self):
        raise NotImplementedError
        linear_vel, angular_vel = 0.1, 0.1
        return np.array([linear_vel, angular_vel])

    def set_gripper_pose(self, gripper_desired_pos_normalized):
        for airbot in self.airbot_players:
            airbot.set_end_effector_value(gripper_desired_pos_normalized)

    def _reset_joints(self):
        if self.reset_position is None:
            reset_position = [robot.default_joints[:6] for robot in self.airbot_players]
        else:
            reset_position = [self.reset_position[:6]] * self.robot_num
        move_arms(self.airbot_players, reset_position, move_time=1)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close."""
        if self.reset_position is None:
            # move_grippers(self.airbot_players, self.eefs_open / 2, move_time=0.5)
            move_grippers(self.airbot_players, self.eefs_open, move_time=1)
        else:
            move_grippers(
                self.airbot_players,
                [self.reset_position[6]] * self.robot_num,
                move_time=1,
            )

    def get_observation(self, get_tracer_vel=False):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        # obs['qvel'] = self.get_qvel()
        # obs['effort'] = self.get_effort()
        obs["images"] = self.get_images()
        if self.use_base:
            obs["base_vel"] = self.get_base_vel()
        if get_tracer_vel:
            obs["tracer_vel"] = self.get_tracer_vel()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        if not fake:
            self._reset_joints()
            self._reset_gripper()
            time.sleep(sleep_time)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(
        self,
        action,
        base_action=None,
        get_tracer_vel=False,
        get_obs=True,
        sleep_time=0,
        arm_vel=0,
    ):
        action = action
        use_planning = False
        for index, robot in enumerate(self.airbot_players):
            jn = robot.all_joints_num
            # robot.set_joint_position_target(
            #     action[jn * index : jn * (index + 1)], [arm_vel], use_planning,
            # )
        time.sleep(sleep_time)
        if base_action is not None:
            raise NotImplementedError
            # linear_vel_limit = 1.5
            # angular_vel_limit = 1.5
            # base_action_linear = np.clip(base_action[0], -linear_vel_limit, linear_vel_limit)
            # base_action_angular = np.clip(base_action[1], -angular_vel_limit, angular_vel_limit)
            base_action_linear, base_action_angular = base_action
            self.tracer.SetMotionCommand(
                linear_vel=base_action_linear, angular_vel=base_action_angular
            )
        # time.sleep(DT)
        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )


def get_arm_joint_positions(bot: AssembledRobot):
    return bot.get_current_joint_positions()[:6]


def get_arm_gripper_positions(bot: AssembledRobot):
    return bot.get_current_joint_positions()[6]


def move_arms(bot_list: List[AssembledRobot], target_pose_list, move_time=1):
    DT = max([bot.dt for bot in bot_list])
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    # 进行关节插值，保证平稳运动
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            # blocking为False用于多台臂可以同时移动
            bot.set_joint_position_target(traj_list[bot_id][t], [6], blocking=False)
        time.sleep(DT)


def move_grippers(bot_list: List[AssembledRobot], target_pose_list, move_time):
    DT = max([bot.dt for bot in bot_list])
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.set_end_effector_value(traj_list[bot_id][t])
        time.sleep(DT)


def make_env(
    setup_robots=True,
    setup_base=False,
    record_images=True,
    robot_instance=None,
    cameras=None,
):
    env = RealEnv(setup_robots, setup_base, record_images, robot_instance, cameras)
    return env


def get_image(ts: dm_env.TimeStep, camera_names, mode=0):
    if mode == 0:  # 输出拼接之后的张量图
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    else:  # 输出独立的张量图（且每个是多维的）  # TODO: 修改为每个是一维的
        curr_image = {}
        for cam_name in camera_names:
            raw_img = ts.observation["images"][cam_name]
            # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow(cam_name, raw_img.astype(np.uint8))
            # cv2.waitKey(0)
            curr_image[cam_name] = torch.from_numpy(
                np.copy(raw_img[np.newaxis, :])
            ).float()
    return curr_image
