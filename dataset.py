import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 自定义数据集类
class BoatTrackDataset(Dataset):
    def __init__(self, gt_path, ob_path, dtype):
        """
        coordinates: 包含 N 个 (2, T) NumPy 数组的列表
        labels: 标签列表（可选）
        transform: 数据变换函数（可选）
        """
        # 将 NumPy 数组列表转换为 PyTorch 张量，形状: [N, 2, T]
        gt_list = np.load(gt_path)
        ob_list = np.load(ob_path)
        assert len(gt_list) == len(ob_list)
        self.coordinates_gt = torch.tensor(np.stack(gt_list, axis=0), dtype=dtype)
        self.coordinates_ob = torch.tensor(np.stack(ob_list, axis=0), dtype=dtype)

    def __len__(self):
        return self.coordinates_gt.shape[0]

    def __getitem__(self, idx):
        gt = self.coordinates_gt[idx, :, :]
        ob = self.coordinates_ob[idx, :, :]
        assert gt.shape == ob.shape
        return gt, ob


def get_dataset(config, type):
    data = BoatTrackDataset(config.gt_path, config.ob_path, config.dtype)

    if type == 'train':
        data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers, pin_memory=True)
    elif type == 'val':
        data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=config.num_workers, pin_memory=True)
    elif type == 'test':
        NotImplementedError('not implemented for teste: {}!'.format(type))
    else:
        raise NotImplementedError('not implemented for this mode: {}!'.format(type))

    return data_loader
