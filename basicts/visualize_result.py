import os

import fire
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.visualize import plot_line
from utils.result import read_result_file
import torch

def visualize_amplitude(file_dir: str):
    # note 先读取三个文件，转换为torch格式
    inputs, target, prediction = read_result_file(file_dir)
    time_steps, window_length, count_variates, count_features = inputs.shape
    frequencies = time_steps // 2 + 1
    print("Changing Features into Fourier Representation")
    amplitude = []
    for s in tqdm(range(count_variates)):
        fft_repr = np.fft.rfft(inputs[:, 0, s, 0], norm="ortho")
        amplitude.append(fft_repr.real**2+fft_repr.imag**2)
        print(amplitude[-1])
    # 生成随机数据模拟振幅
    # 数据形状为 (时间步, 频率)
    amplitude = np.stack(amplitude, axis=0)
    print(amplitude.shape)

    # 创建时间步和频率的网格
    variates = np.arange(count_variates)
    freq = np.arange(frequencies)

    # 绘制频谱图
    plt.figure(figsize=(24, 6))
    plt.pcolormesh(variates, freq, amplitude.T, shading='auto', cmap='viridis')

    # 添加颜色条
    plt.colorbar(label='Amplitude')

    # 设置坐标轴标签
    plt.xlabel('Variates')
    plt.ylabel('Frequency')
    plt.tight_layout()
    # 显示图形
    fig_dir = os.path.join(os.path.dirname(file_dir), "visualize_result")
    os.makedirs(fig_dir, exist_ok=True)
    fig_name = os.path.join(fig_dir, "amplitude.png")
    plt.savefig(fig_name)

def visualize_prediction(file_dir: str,
                         sensor_id: int = 0,
                         horizon: int = 11,):
    inputs, target, prediction = read_result_file(file_dir)
    data = target[:, horizon, sensor_id, 0]
    pred = prediction[:, horizon, sensor_id, 0]
    x = np.arange(data.shape[0])
    # 输出位置
    fig_dir = os.path.join(os.path.dirname(file_dir), "visualize_result")
    os.makedirs(fig_dir, exist_ok=True)
    # 开始绘图
    plot_line(base_folder=fig_dir,
              title=f"Visualization of Sensor {sensor_id} on Horizon {horizon+1}",
              x=x, ys=[data, pred, np.abs(data-pred)], y_names=["Label", 'Prediction', 'error'],
              x_axis_name="time steps", y_axis_name="value",
              y_min=min(data), y_max=max(data))

def visualize_mean_prediction(file_dir: str,
                              sensor_id: int = 0,
                              ):
    inputs, target, prediction = read_result_file(file_dir)
    data = np.mean(target[:, :, sensor_id, 0], axis=1)
    pred = np.mean(prediction[:, :, sensor_id, 0], axis=1)
    x = np.arange(data.shape[0])
    # 输出位置
    fig_dir = os.path.join(os.path.dirname(file_dir), "visualize_result")
    os.makedirs(fig_dir, exist_ok=True)
    # 开始绘图
    plot_line(base_folder=fig_dir,
              title=f"Visualization of Sensor {sensor_id}'s Mean Prediction",
              x=x, ys=[data, pred], y_names=["Label", 'Prediction'],
              x_axis_name="time steps", y_axis_name="value",
              y_min=min(data), y_max=max(data))

def visualize_trend_prediction(file_dir: str,
                              sensor_id: int = 0,
                              ):
    inputs, target, prediction = read_result_file(file_dir)
    label_stds = []
    pred_stds = []
    x_a = np.array([float(i) for i in range(prediction.shape[1])])
    for i in tqdm(range(inputs.shape[0])):
        data = target[i, :, sensor_id, 0]
        data_trend = np.polyfit(x_a, data, 1)
        data -= (x_a*data_trend[0] + data_trend[1])
        label_stds.append(data_trend[0])
        pred = prediction[i, :, sensor_id, 0]
        pred_trend = np.polyfit(x_a, pred, 1)
        pred -= (x_a*pred_trend[0] + pred_trend[1])
        pred_stds.append(pred_trend[0])
    label_stds = np.array(label_stds)
    pred_stds = np.array(pred_stds)
    x = np.arange(target.shape[0])
    # 输出位置
    fig_dir = os.path.join(os.path.dirname(file_dir), "visualize_result")
    os.makedirs(fig_dir, exist_ok=True)
    # 开始绘图
    plot_line(base_folder=fig_dir,
              title=f"Visualization of Sensor {sensor_id}'s Prediction's Trend",
              x=x, ys=[label_stds, pred_stds], y_names=["Label", 'Prediction'],
              x_axis_name="time steps", y_axis_name="value",
              y_min=min(label_stds), y_max=max(label_stds))


if __name__ == "__main__":
    fire.Fire()