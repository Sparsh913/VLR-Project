import numpy as np
import os
import re
from PIL import Image
import time

def gripper(path):
    # find .npz files in path_dataset through regex and store it in a list. the episodes name are in this format: episode_0358482.npz
    episodes = [epi for epi in os.listdir(path) if re.match(r'episode_[0-9]+.npz', epi)]
    print("Num of episodes: ", len(episodes))
    print(episodes[0])
    print(type(episodes[0]))
    for epi in episodes:
        data = np.load(path + epi)
        gripper_width, gripper_action = data['robot_obs'][6], data['robot_obs'][-1]
        print(f"gripper width: {gripper_width}, gripper action: {gripper_action}")
        # sleep for 1 second
        time.sleep(1)
    
if __name__ == "__main__":
    path = "./calvin_debug_dataset/training/"
    gripper(path)