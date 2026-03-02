import torch
from modelscope.models.cv.video_human_matting import preprocess
from torch.utils.data import Dataset
from Scheduler import Scheduler
import json
import pickle

class VLADataset(Dataset):
    def __init__(self, json_dir, trajectory_dir):
        super().__init__()
        with open(json_dir, 'r') as f:
            self.vqa_data = json.dumps(f)

        f.close()

        with open(trajectory_dir, 'r') as f:
            self.trajectory_data = torch.from_numpy(pickle.dumps(f))

        f.close()

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, index):
        data ={'message': self.vqa_data[index], 'gt_trajectory': self.trajectory_data[index]}
        return data