import os
import time
import torch
from utils import plot_example
import random

class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.model.cuda()

        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_state')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if self.config.type == 'predict':
            self.pre_length = self.config.window_length

    def train(self, dataloader, train_log, global_step):
        self.model.train()
        start = time.time()

        for idx, (gt, ob) in enumerate(dataloader):
            gt = gt.cuda()
            ob = ob.cuda()
            if self.config.type == 'filter':
                if self.config.window_type == 'sliding':
                    gt, ob = self.random_window(gt, ob)
                elif self.config.window_type == 'fix':
                    widx = random.choice([0, 1]) * self.config.window_length
                    gt, ob = self.fix_window(gt, ob, window_idx=widx)
                else:
                    raise NotImplementedError
            elif self.config.type == 'predict':
                gt, ob = self.pre_window(gt, ob, window_idx=0)
            data_train = torch.cat((gt, ob), dim=0)
            normalizer = MinMaxNormalizer(data_train)
            ob = normalizer.normalize(ob)
            if self.config.type == 'predict':
                gt = gt[:, :, self.config.window_length:self.config.window_length+self.pre_length]
                ob = ob[:, :, 0:self.config.window_length]

            predict_track = self.model(ob)
            predict_track = normalizer.denormalize(predict_track)

            loss = self.loss_fn(predict_track, gt)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            db_loss = 10 * torch.log10(loss)

            global_step += 1
            if os.path.exists(train_log):
                with open(train_log, 'a') as f:
                    f.write('time' + str(time) + 'step' + str(global_step) + 'loss' + str(db_loss) + 'db' + '\n')
            else:
                with open(train_log, 'w') as f:
                    f.write('time' + str(time) + 'step' + str(global_step) + 'loss' + str(db_loss) + 'db' + '\n')
        return global_step

    def validate(self, dataloader, val_log, epoch):
        self.model.eval()

        total_loss = []
        with torch.no_grad():
            for idx, (gt, ob) in enumerate(dataloader):
                gt = gt.cuda()
                ob = ob.cuda()
                if self.config.type == 'filter':
                    if self.config.window_type == 'sliding':
                        gt, ob = self.random_window(gt, ob)
                    elif self.config.window_type == 'fix':
                        widx = random.choice([0, 1]) * self.config.window_length
                        gt, ob = self.fix_window(gt, ob, window_idx=widx)
                    else:
                        raise NotImplementedError
                elif self.config.type == 'predict':
                    gt, ob = self.pre_window(gt, ob, window_idx=0)

                else:
                    print(self.config.type)
                    raise NotImplementedError
                data_val = torch.cat((gt, ob), dim=0)
                normalizer = MinMaxNormalizer(data_val)
                ob = normalizer.normalize(ob)
                gt_plot = gt
                if self.config.type == 'predict':
                    gt = gt[:, :, self.config.window_length:self.config.window_length + self.pre_length]
                    ob = ob[:, :, 0:self.config.window_length]


                predict_track = self.model(ob)
                predict_track = normalizer.denormalize(predict_track)
                val_loss = self.loss_fn(predict_track, gt)
                total_loss.append(val_loss)

                if idx == len(dataloader) - 1:
                    if self.config.save_val_example:
                        img_dir = self.config.save_dir + "/val_img"
                        os.makedirs(img_dir, exist_ok=True)
                        img_path = os.path.join(img_dir, f'val_{epoch}.png')
                        ob_plt = normalizer.denormalize(ob)
                        sample_idx = random.randint(a=0, b=gt.shape[0]-1)
                        plot_example(gt_plot[sample_idx, :, :], ob_plt[sample_idx, :, :], predict_track[sample_idx, :, :], img_path)




        stacked_tensor = torch.stack(total_loss, dim=0)
        mean_loss = torch.mean(stacked_tensor, dim=0)
        db_loss = 10 * torch.log10(mean_loss)
        if os.path.exists(val_log):
            with open(val_log, 'a') as f:
                f.write('epoch' + str(epoch) + 'loss' + str(db_loss) + 'db' + '\n')
        else:
            with open(val_log, 'w') as f:
                f.write('epoch' + str(epoch) + 'loss' + str(db_loss) + 'db' + '\n')
        return mean_loss , db_loss

    def random_window(self, gt, ob):
        assert gt.shape[0] == ob.shape[0] == self.config.batch_size
        assert gt.shape[1] == ob.shape[1] == self.config.in_channel
        window_idx = torch.randint(0, gt.shape[2] - self.config.window_length, (1,)).item()
        gt_cut = gt[:, :, window_idx:window_idx + self.config.window_length]
        ob_cut = ob[:, :, window_idx:window_idx + self.config.window_length]
        return gt_cut, ob_cut

    def fix_window(self, gt, ob, window_idx):
        assert gt.shape[0] == ob.shape[0] == self.config.batch_size
        assert gt.shape[1] == ob.shape[1] == self.config.in_channel
        gt_cut = gt[:, :, window_idx:window_idx + self.config.window_length]
        ob_cut = ob[:, :, window_idx:window_idx + self.config.window_length]
        return gt_cut, ob_cut

    def pre_window(self, gt, ob, window_idx):
        assert gt.shape[0] == ob.shape[0] == self.config.batch_size
        assert gt.shape[1] == ob.shape[1] == self.config.in_channel
        gt_cut = gt[:, :, window_idx:window_idx + self.config.window_length+self.pre_length]
        ob_cut = ob[:, :, window_idx:window_idx + self.config.window_length+self.pre_length]
        return gt_cut, ob_cut

    def save_ckpt(self, epoch):
        state_dict = {'epoch': epoch,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      }
        torch.save(state_dict, self.checkpoint_path + f'/model_best{epoch}.pt')

    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        last_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {ckpt_path}, resuming at epoch {last_epoch}")


class MinMaxNormalizer:
    def __init__(self, data):
        """
        data: 输入张量，任意形状 [N, 2, T]
        """
        self.data = data
        # 计算最小值和最大值（沿着所有维度）
        self.min_vals = data.min()  # 全局最小值
        self.max_vals = data.max()  # 全局最大值
        self.range = self.max_vals - self.min_vals + 1e-8  # 避免除以零

    def normalize(self, input_data):
        """
        将数据归一化到 [0, 1]
        """
        normalized_data = (input_data - self.min_vals) / self.range
        return normalized_data

    def denormalize(self, normalized_data):
        """
        将归一化数据恢复到原始范围
        """
        original_data = normalized_data * self.range + self.min_vals
        return original_data

