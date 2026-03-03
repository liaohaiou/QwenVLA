import numpy as np
import torch
from modelscope.models.cv.video_human_matting import preprocess
from torch.utils.data import Dataset

from QwenVLA.QwenVLABenchDriveMessage import trajectory
from Scheduler import Scheduler
import json
import pickle

from qwen_vl_benchdrive import message


class VLADataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.messages = []
        self.trajectories = []
        with open(dataset_dir, 'r') as f:
            dataset = json.load(f)
            for data in dataset:
                content = data[0 : 7]
                trajectory = data[7]["trajectory"]
                message = [{  "role": "user", "content" : content}]
                self.messages.append(message)
                trajectory_tensor = torch.from_numpy(np.array(trajectory))
                self.trajectories.append(trajectory_tensor)

        f.close()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        data ={'message': self.messages[index], 'gt_trajectory': self.trajectories[index]}
        return data