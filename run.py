import subprocess

# Define the command
command = "python tools/infer.py --weights yolov6s_mbla.pt --use_depth_cam --webcam-addr 0 --view-img"

# Use subprocess to run the command in the terminal
subprocess.run(command, shell=True)
