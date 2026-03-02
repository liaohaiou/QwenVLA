import configparser
import argparse
import torch
from QwenDriveVLA import QwenDriveVLA
from Scheduler import  Scheduler
import torch.nn as nn
from VLADataset import VLADataset
from torch.utils.data import DataLoader
import torch.optim as optim

from qwen_vl_benchdrive import message


def readconfig(config_file) :
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    cfg = {}
    cfg['llm_path'] = config_parser['llm_path']
    cfg['adapter_ckpt'] = config_parser['adapter_ckpt']
    cfg['d_model'] = config_parser['d_model']
    cfg['timesteps'] = config_parser['timesteps']
    cfg['offset'] = config_parser['offset']
    cfg['traj_cordinate_dim'] = config_parser['traj_cordinate_dim']
    cfg['num_heads'] = config_parser['num_heads']
    cfg['num_layers'] = config_parser['num_layers']
    cfg['vqa_dir'] = config_parser['vqa_dir']
    cfg['trajectory_dir'] = config_parser['trajectory_dir']
    cfg['epoches'] = config_parser['epoches']
    return  cfg

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str)
    args = arg_parser.parse_args()
    cfg = readconfig(args.config)
    vqa_dir = cfg['vqa_dir']
    trajectory_dir = cfg['trajectory_dir']
    epoches = cfg['epoches']

    qwen_drive_vla = QwenDriveVLA(cfg)
    qwen_drive_vla.to(device='cuda', dtype=torch.float16)
    scheduler = Scheduler(cfg)
    criterion = nn.MSELoss()
    train_dataset = VLADataset(vqa_dir, trajectory_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    optimizer = optim.Adam(qwen_drive_vla.parameters(), lr=0.0001)
    for i in range(epoches):
        for data in train_dataloader:
            message = data['message']
            gt_trajectory = data['gt_trajectory'].to(device='cuda', dtype=torch.float16)
            timestep = torch.randint(0, 50, (1,))
            noisy_trajectory, eps_gt = scheduler.add_noise(gt_trajectory, timestep)
            eps_gt = eps_gt.to(dtype=torch.float16, device='cuda')
            eps_predicted = qwen_drive_vla(timestep, noisy_trajectory, message).to(dtype=torch.float16, device='cuda')
            loss = criterion(eps_gt, eps_predicted)
            optimizer.zero_grad()
            loss.back_ward()
            optimizer.step()