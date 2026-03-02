from trl.trainer.utils import forward

from QwenVL import QwenVL
from TrajectoryAction import ActionHead, Scheduler
import torch
import torch.nn as nn

class QwenDriveVLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scheduler = Scheduler(cfg)
        self.qwen_vl = QwenVL(cfg)
        self.action_head = ActionHead(cfg)
        self.freeze_vlm = cfg['freeze_vlm']
        if self.freeze_vlm:
            for p in self.qwen_vl.parameters():
                p.requires_grad = False



    def forward(self, timestamp, traj_noise, images ):
        if self.freeze_vlm:
            with torch.no_grad():
                condition_embedding = self.qwen_vl(images)
        else:
            condition_embedding = self.qwen_vl(images)
        eps = self.action_head(traj_noise, timestamp, condition_embedding)
        return eps



