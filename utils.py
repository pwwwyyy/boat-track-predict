import torch
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_example(gt, ob, pre, path):
    # assert gt.shape == ob.shape == pre.shape
    assert gt.ndim == 2
    lat_gt, lon_gt = gt.cpu().numpy()
    lat_ob, lon_ob = ob.cpu().numpy()
    lat_pre, lon_pre = pre.cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.plot(lat_gt, lon_gt, 'o-', label='gt', color='r', linewidth=2)
    plt.plot(lat_ob, lon_ob, 's-', label='ob', color='b', linewidth=2)
    plt.plot(lat_pre, lon_pre, '^-', label='pre', color='g', linewidth=2)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.legend()

    plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()
