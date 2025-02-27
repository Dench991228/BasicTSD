import os

import fire
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.visualize import plot_line, plot_mesh
from utils.result import read_result_file
import torch

def visualize_amplitude(file_dir: str, sensor_id: int = 0):
    # note 先读取三个文件，转换为torch格式
    inputs, target, prediction = read_result_file(file_dir)
    time_steps, window_length, count_variates, count_features = inputs.shape
    frequencies = time_steps // 2 + 1
    print("Changing Features into Fourier Representation")
    amplitude = []
    top5percent = int(frequencies * 0.05)
    inputs[:, 0, sensor_id, 0] -= np.mean(inputs[:, 0, sensor_id, 0])
    fft_repr = np.fft.rfft(inputs[:, 0, sensor_id, 0], norm="ortho")
    amplitude.append(np.sqrt(fft_repr.real**2+fft_repr.imag**2))
    print(amplitude[-1])
    sorted_amplitude = np.argsort(amplitude[-1])
    print(sorted_amplitude[-top5percent:])
    # 生成随机数据模拟振幅
    # 数据形状为 (时间步, 频率)
    amplitude = np.stack(amplitude, axis=0)
    print(amplitude.shape)
    # 显示图形
    fig_dir = os.path.join(os.path.dirname(file_dir), "visualize_result")
    plot_line(base_folder=fig_dir,
              title="Amplitude of last sensor",
              x_axis_name="k",
              x=[i for i in range(frequencies)],
              ys=[amplitude[-1]],
              y_names=["amplitude of last sensor"],
              y_axis_name="amplitude",)

def visualize_amplitude_change(file_dir: str, sensor_id: int = 0):
    """
    展现某个节点频率分量随时间变化
    :param file_dir: 目标的预测文件
    :param sensor_id: 目标的sensor_id
    """
    inputs, target, prediction = read_result_file(file_dir)
    time_steps, window_length, count_variates, count_features = inputs.shape
    frequencies = window_length // 2 + 1
    print("Changing Features into Fourier Representation")
    # 先弄输入特征
    # (count_timesteps, window_length)
    amplitude_repr = []
    for arr in [target, prediction]:
        inputs = arr[:, :, sensor_id, 0]
        amplitude_inputs = []
        for i in range(time_steps):
            inputs[i] -= np.mean(inputs[i])
            fft_repr = np.fft.rfft(inputs[i], norm="ortho")
            amplitude_inputs.append(np.sqrt(fft_repr.real**2+fft_repr.imag**2))
        amplitude_inputs = np.stack(amplitude_inputs, axis=0)
        amplitude_repr.append(amplitude_inputs)
    amplitude_repr.append(np.abs(amplitude_repr[-1] - amplitude_repr[-2]))

    for arr, name in zip(amplitude_repr, ["label", "pred", "amplitude error"]):
        fig_dir = os.path.join(os.path.dirname(file_dir), "visualize_result")
        plot_mesh(fig_dir, f"Distribution of {name}s' frequency components through time", arr, 150)

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