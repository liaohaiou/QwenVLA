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
    cfg['llm_path'] = config_parser["Train"]['llm_path']
    cfg['adapter_ckpt'] = config_parser["Train"]['adapter_ckpt']
    cfg['d_model'] = config_parser.getint("Train", 'd_model')
    cfg['timesteps'] = config_parser.getint("Train", 'timesteps')
    cfg['offset'] = config_parser.getfloat("Train", "offset")
    cfg['traj_cordinate_dim'] = config_parser.getint("Train", 'traj_cordinate_dim')
    cfg['num_heads'] = config_parser.getint("Train", 'num_heads')
    cfg['num_layers'] = config_parser.getint("Train", 'num_layers')
    cfg['dataset'] = config_parser["Train"]['dataset']
    cfg['epoches'] = config_parser.getint("Train", 'epoches')
    cfg['freeze_vlm'] = config_parser.getboolean("Train", 'freeze_vlm')
    return  cfg

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str)
    args = arg_parser.parse_args()
    config_file = '/home/huangweihao/swift/QwenVLA/train_config.ini'
    cfg = readconfig(config_file)
    dataset = cfg['dataset']
    epoches = cfg['epoches']

    qwen_drive_vla = QwenDriveVLA(cfg)
    qwen_drive_vla.to(device='cuda', dtype=torch.float16)
    scheduler = Scheduler(cfg)
    criterion = nn.MSELoss()
    train_dataset = VLADataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    optimizer = optim.Adam(qwen_drive_vla.parameters(), lr=0.0001)
    for i in range(epoches):
        for data in train_dataloader:
            message = data['message']
            gt_trajectory = data['gt_trajectory'].to(device='cuda', dtype=torch.float16)
            timestep = torch.randint(0, 50, (1,))
            noisy_trajectory, eps_gt = scheduler.add_noise(gt_trajectory, timestep)
            eps_gt = eps_gt.to(dtype=torch.float16, device='cuda')
            timestep.to(device='cuda', dtype=torch.float16)
            eps_predicted = qwen_drive_vla(timestep, noisy_trajectory, message).to(dtype=torch.float16, device='cuda')
            loss = criterion(eps_gt, eps_predicted)
            optimizer.zero_grad()
            loss.back_ward()
            optimizer.step()