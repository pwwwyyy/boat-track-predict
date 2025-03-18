import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# python generate_data.py

EARTH_RADIUS = 6371000

np.random.seed(42)

def generate_track(start_speed, start_heading, motion_type, SNR_track,
                   acceleration=None, turn_rate=None, num_points=200):
    """
    生成航迹数据
    :param start_time: 起始时间（datetime对象）
    :param start_lat: 起始纬度（度）
    :param start_lon: 起始经度（度）
    :param start_speed: 起始速度（米/秒）
    :param start_heading: 起始航向（度，0表示正北，90表示正东）
    :param motion_type: 运动类型（'uniform'：匀速，'acceleration'：匀加速，'turn'：转弯）
    :param num_points: 航迹点数
    :return: 包含时间、经纬度、航速和航向的DataFrame
    """
    lat = np.zeros(num_points)
    lon = np.zeros(num_points)
    speed = np.zeros(num_points)
    heading = np.zeros(num_points)

    lat[0] = 0
    lon[0] = 0
    speed[0] = start_speed
    heading[0] = start_heading

    for i in range(1, num_points):
        dt = 1

        if motion_type == 'uniform':
            # 匀速运动
            speed[i] = start_speed
            heading[i] = start_heading
        elif motion_type == 'acceleration':
            # 匀加速运动，假设加速度为0.1 m/s^2
            speed[i] = speed[i - 1] + acceleration * dt
            heading[i] = start_heading
        elif motion_type == 'turn':
            # 转弯运动，假设转弯速率为1度/秒
            speed[i] = start_speed
            heading[i] = heading[i - 1] + turn_rate * dt

        # 计算经纬度变化
        distance = speed[i] * dt
        angular_distance = distance / EARTH_RADIUS

        # 将航向转换为弧度
        heading_rad = np.radians(heading[i])

        # 计算新的纬度
        lat[i] = lat[i - 1] + np.degrees(angular_distance * np.cos(heading_rad))

        # 计算新的经度
        lon[i] = lon[i - 1] + np.degrees(angular_distance * np.sin(heading_rad) / np.cos(np.radians(lat[i - 1])))

    gt_track = np.stack((lat, lon), axis=0)
    noised_lat = add_awgn(lat, SNR_track)
    noised_lon = add_awgn(lon, SNR_track)
    ob_track = np.stack((noised_lat, noised_lon), axis=0)

    return gt_track, ob_track


def add_awgn(signal, snr_db):
    """
    为输入信号添加加性高斯白噪声（AWGN）。

    参数：
        signal: 输入信号（NumPy 数组）
        snr_db: 信噪比（以分贝为单位）

    返回：
        受噪声影响的信号
    """
    # 计算信号的功率
    signal_power = np.mean(signal ** 2)

    # 将信噪比从分贝转换为线性
    snr_linear = 10 ** (snr_db / 10.0)

    # 计算噪声的功率
    noise_power = signal_power / snr_linear

    # 生成噪声
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    # 返回添加了 AWGN 的信号
    return signal + noise


def plot_sample(gt, ob):
    latitudes1, longitudes1 = gt
    latitudes2, longitudes2 = ob
    plt.figure(figsize=(10, 6))
    plt.plot(longitudes1, latitudes1, c='blue', marker='o', label='gt')
    plt.plot(longitudes2, latitudes2, c='red', marker='x', label='ob')
    plt.show()


N_tracks = 16384
num_points = 256
SNR_track = 30
start_speed = 30
start_heading = 45
acceleration = 0.5
turn_rate = 0.3

gt = []
ob = []
for i in tqdm(range(N_tracks)):
    # 生成均匀航迹
    gt_u, ob_u = generate_track(start_speed, start_heading, motion_type="uniform",
                                SNR_track=SNR_track, acceleration=None, turn_rate=None,
                                num_points=num_points)
    gt.append(gt_u)
    ob.append(ob_u)
    gt_a, ob_a = generate_track(start_speed, start_heading, motion_type="acceleration",
                                SNR_track=SNR_track, acceleration=acceleration,
                                turn_rate=None, num_points=num_points)
    gt.append(gt_a)
    ob.append(ob_a)
    gt_t, ob_t = generate_track(start_speed, start_heading, motion_type="turn", SNR_track=SNR_track,
                                acceleration=None, turn_rate=turn_rate, num_points=num_points)
    gt.append(gt_t)
    ob.append(ob_t)

print("num of ground truth" , len(gt))
print("num of observation" , len(ob))

np.save("./data/gt.npy", gt)
np.save("./data/ob.npy", ob)


