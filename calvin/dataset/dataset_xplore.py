import numpy as np
import os
import re
from PIL import Image

def extract_data(path_dataset):
    # find .npz files in path_dataset through regex and store it in a list. the episodes name are in this format: episode_0358482.npz
    episodes = [epi for epi in os.listdir(path_dataset) if re.match(r'episode_[0-9]+.npz', epi)]
    print("Num of episodes: ", len(episodes))
    epi = episodes[510]
    # epi = "episode_0360950.npz"
    # print(type(episodes))
    # Load the first episode
    data = np.load(path_dataset + epi)
    # If a directory with the episode name already exists, delete it
    if os.path.exists(epi):
        os.system("rm -rf " + epi)
    # make a directory with the episode name if it doesn't exist already
    if not os.path.exists(epi):
        os.makedirs(epi)
    output_path = "./" + epi + "/"
    # extract the data from the episode and save all the data as npy files
    for key, value in data.items():
        np.save(output_path + f"{key}.npy", value)
        print(f"saved {key}.npy")
        print(f"shape of {key}.npy is {value.shape}")
        
    # Convert .npy images to .jpg images
    for img in os.listdir(output_path):
        # check if the file name starts either with 'rgb' or 'depth'
        if not re.match(r'(rgb|depth).*', img):
            continue
        print(img)
        img_array = np.load(output_path + img)
        # img_array = np.squeeze(img_array)
        print("shape after loading: ", img_array.shape)
        # if img.startswith("rgb"):
            # img_array = np.transpose(img_array, (2, 0, 1))
            # img_array = np.full((img_array.shape[0], img_array.shape[1], 3), 255, dtype=np.uint8)
        #     print("shape after transpose: ", img_array.shape)
            
        # img_array = img_array * 255
        # normalize
        # img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        # print("shape after normalization: ", img_array.shape)
        # img_array = np.uint8(img_array)
        # print("shape after uint8: ", img_array.shape)
        if img.startswith("rgb"):
            img_array = Image.fromarray(img_array, mode='RGB')
        else:
            try:
                img_array = Image.fromarray(img_array[:,:,0], mode='L')
            except:
                img_array = Image.fromarray(img_array, mode='L')
        print("shape after Image.fromarray: ", img_array.size)
        # save the image
        img_array.save(output_path + img[:-4] + ".jpg")
        print(f"saved {img[:-4]}.jpg")
        

if __name__ == "__main__":
    path_dataset = "./calvin_debug_dataset/training/"
    extract_data(path_dataset)