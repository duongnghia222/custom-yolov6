import subprocess

# Define the command
# command = "python tools/infer.py --weights yolov6s_mbla.pt --use_depth_cam --view-img --not-save-img"
command = "python tools/infer.py --weights best_ckpt.pt --use_depth_cam --view-img --not-save-img --yaml data/data.yaml"

# Use subprocess to run the command in the terminal
subprocess.run(command, shell=True)
