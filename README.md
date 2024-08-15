# Instruction for OpenVLA with Airbot
> For any questions, please contact dkr21@mails.tsinghua.edu.cn or leave a github issue.

## Data Collection and Pre-Processing

- cd act_data
- perform
> airbot_demonstrate -mts 100 -cm 0 -tn Stack_cups -f 10 --raw-video -se 0 --master-end-mode teacherv2
-mts: max time steps
-cm: camera, use 0 for 1 camera
-tn: task name
-f: frequency
--raw-video: whether save raw video
-se: start episode, use n if you have already sampled n videos
--master-end-mode: which teacher do you use

> python ./convert_episodes.py -rn 1 -cn 0 -tn Stack_cups -se 0 -ee 100 -rd ./demonstrations -mp 3 -or eef_pose
-rn: robot number, use 1
-cn: camera name, use 0
-tn: task name
-se: start episode, use 0 for default
-ee: end episode
-rd: root directory
-mp: multi-processing
-or: other parameters, please keep this


## Convert Dataset to rlds format

- install rlds following https://github.com/kpertsch/rlds_dataset_builder
- replace and cd rlds/airbot_mix
- change VERSION and RELEASE_NOTES in airbot_mix_dataset_builder.py
- python airbot_mix_dataset_builder.py
- tfds build


## Finetune OpenVLA

- Install Openvla following https://github.com/openvla/openvla
- Replace prismatic and vla-scripts with what we provide
- copy tfds dataset from ~/tensorflow_datasets/airbot_mix to openvla/data
- run sh vla-scripts/finetune.sh
- finish training
- Deploy by python vla-scripts/deploy.py


## Deploy on airbot

- run
> python vla-scripts/airbot_client.py -cmd "Put the mark into the bowl." -cm 0 -uk airbot_mix -can can1 --time_steps 40

The meaning for the parameters are clear.