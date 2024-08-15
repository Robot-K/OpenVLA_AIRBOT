airbot_demonstrate -mts 400 -cm 0 -cm 1 -cm 2 -cm 3 -cm 4 -cm 5 -cw 2 -tn kding_greenbkg_50 -f 30 --raw-video -se 0
python ./convert_episodes.py -rn 1 -cn 1 -tn kding_greenbkg_50 -se 0 -ee 50 -rd ./demonstrations -mp 3


airbot_demonstrate -mts 100 -cm 0 -tn Mark -f 10 --raw-video -se 0 --master-end-mode teacherv2
python ./convert_episodes.py -rn 1 -cn 0 -tn Mark -se 0 -ee 40 -rd ./demonstrations -mp 3 -or eef_pose

airbot_demonstrate -mts 100 -cm 0 -cm 1 -tn Stack_cups -f 10 --raw-video -se 0 --master-end-mode teacherv2
python ./convert_episodes.py -rn 1 -cn 0,1 -tn Stack_cups -se 0 -ee 40 -rd ./demonstrations -mp 3 -or eef_pose


airbot_demonstrate -mts 100 -cm 0 -cm 1 -tn Stack_cups -f 10 --raw-video -se 27 --master-end-mode teacherv2
python3 ./convert_episodes.py -rn 1 -cn 0,1 -tn Stack_cups -se 0 -ee 100 -rd ./demonstrations -mp 3 -or eef_pose