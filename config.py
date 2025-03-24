import configparser
import torch
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Config:
    def __init__(self, config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)

        # experiment
        self.seed = int(parser.get('experiment', 'seed'))
        self.exp_name = parser.get('experiment', 'exp_name')

        # training
        self.gt_path = parser.get('training', 'gt_path')
        self.ob_path = parser.get('training', 'ob_path')
        self.batch_size = int(parser.get('training', 'batch_size'))

        self.gpu = parser.get('training', 'gpu')
        self.window_length = int(parser.get('training', 'window_length'))
        self.window_type = parser.get('training', 'window_type')
        self.num_workers = int(parser.get('training', 'num_workers'))
        self.num_epochs = int(parser.get('training', 'num_epochs'))
        self.lr = float(parser.get('training', 'lr'))
        self.save_dir = parser.get('training', 'save_dir') + '/' + f"{timestamp}"
        self.dtype_str = parser.get('training', 'dtype')

        # model
        self.in_channel = int(parser.get('model', 'in_channel'))
        self.n_channels = int(parser.get('model', 'n_channels'))
        self.n_blocks = int(parser.get('model', 'n_blocks'))

        # val
        self.val_period = int(parser.get('validation', 'val_period'))
        self.save_val_example = bool(parser.get('validation', 'save_val_example'))


        self.dtype_mapping = {
            'float32': torch.float32,
            'float64': torch.float64,
            'float16': torch.float16,
            'bf16': torch.bfloat16,
        }
        self.dtype = self.dtype_mapping[self.dtype_str]