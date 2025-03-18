import os
import torch
import argparse
import random
import numpy as np
from config import Config
from dataset import get_dataset

from unet.unet_1d import UNet_1D
from train import Trainer
from utils import count_parameters


def train(config):
    train_log = config.save_dir + "/train_log.txt"
    val_log = config.save_dir + "/val_log.txt"
    os.makedirs(config.save_dir, exist_ok=True)
    train_dataloader = get_dataset(config, type='train')
    print("training batch num", len(train_dataloader))
    valid_dataloader = get_dataset(config, type='val')
    model = UNet_1D(config.in_channel)
    # model = model.to(dtype=config.dtype)
    trainer = Trainer(config=config, model=model)

    print(f'num parameters: {count_parameters(model)}')

    best_loss = 1000000
    last_epoch = 0
    global_step = 0

    for epoch in range(last_epoch, config.num_epochs):
        global_step = trainer.train(train_dataloader, train_log, global_step)

        if (epoch + 1) % config.val_period == 0 or epoch == config.num_epochs - 1:
            val_loss, db_loss = trainer.validate(valid_dataloader, val_log, epoch + 1)
            print(f"val loss of epoch {epoch + 1} {db_loss} db")
            if val_loss < best_loss:
                best_loss = val_loss
                trainer.save_ckpt(epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='')
    parser.add_argument('--config_path', type=str, default='./boat_track_stimu.cfg', help='path to config file with hyperparameters, etc.')
    args = parser.parse_args()

    config = Config(args.config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if args.train:
        train(config)
