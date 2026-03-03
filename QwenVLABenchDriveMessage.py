from pathlib import Path
import json
import numpy as np
from qwen_vl_benchdrive import rgb_fronts

benchdrive_dataset = '/home/huangweihao/Bench2Drive/Bench2Drive-main/Orion/data/bench2drive/test'
benchdrive_dataset_path = Path(benchdrive_dataset)
messages = []
for town in benchdrive_dataset_path.iterdir():
    camera = town.joinpath('camera')
    rgb_fronts = camera.joinpath('rgb_front')
    rgb_front_lefts = camera.joinpath('rgb_front_left')
    rgb_front_rights = camera.joinpath('rgb_front_right')
    rgb_backs = camera.joinpath('rgb_back')
    rgb_back_lefts = camera.joinpath('rgb_back_left')
    rgb_back_rights = camera.joinpath('rgb_back_right')
    for front_image in rgb_fronts.iterdir():
        message = []
        front_image_name = front_image.name
        front_left_image = rgb_front_lefts.joinpath(front_image_name)
        front_right_image = rgb_front_rights.joinpath(front_image_name)
        back_image = rgb_backs.joinpath(front_image_name)
        back_left_image = rgb_back_lefts.joinpath(front_image_name)
        back_right_image = rgb_back_rights.joinpath(front_image_name)
        image_path_list = [front_image, front_left_image, front_right_image, back_image, back_left_image, back_right_image]
        for item in image_path_list:
            image_message =  {"type": "image", "image": item.__str__()}
            message.append(image_message)

        text_message = {"type" : "text", "text" : "you are driver, there are surronding environmet of the car. Give the driving signal"}
        message.append(text_message)
        rng = np.random.default_rng()
        trajectory = rng.random((6, 2))
        trajectory_message = {"type" : "trajectory", "trajectory" : trajectory.tolist()}
        message.append(trajectory_message)
        messages.append(message)

with open("train_dataset.json", 'w') as f:
    json.dump(messages, f)



