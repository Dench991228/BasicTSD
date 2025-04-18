import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt


def plot_line( title: str,
              x: List|np.ndarray, ys: List, y_names: List,
              base_folder: str = None,
              y_max = None, y_min = None,
              x_axis_name: str = "time steps", y_axis_name: str = "loss",
              w: int = None, no_line: bool = False,
              ):
    """
    绘制折线图的函数
    :param base_folder: 图片的文件夹
    :param title: 图片的题目
    :param x: 横轴
    :param ys: 各个序列
    :param y_names: 各个序列的名称
    :param y_max: y轴最大值
    :param y_min: y轴最小值
    :param x_axis_name: 横轴的名称
    :param y_axis_name: 纵轴的名称
    :param w: 绘制竖线
    """
    # 创建一个新的图形
    plt.figure(figsize=(24, 8))

    # 挨个绘制折线
    for y, y_name in zip(ys, y_names):
        if not no_line:
            plt.plot(x, y, label=y_name, marker='.', linestyle='-', markersize=1, linewidth=4)
        else:
            plt.plot(x, y, label=y_name, marker='.', linestyle='none', markersize=1, linewidth=4)


    # 添加标题
    plt.title(title)
    # 添加X轴和Y轴标签
    plt.xlabel(x_axis_name, fontsize=24)
    plt.ylabel(y_axis_name, fontsize=24)
    plt.ylim(y_min, y_max)
    plt.tick_params(axis='both', labelsize=16)
    # 显示图例
    plt.legend(fontsize=16)
    # 绘制竖线
    if w is not None:
        plt.axvline(x=w,  linestyle='-')

    # 显示网格
    plt.grid(True)
    plt.tight_layout()
    # 显示图形
    if base_folder is not None:
        fig_name = os.path.join(base_folder, f"{title}.png")
        plt.savefig(fig_name)
    else:
        plt.show()


def plot_mesh(title: str,
              amplitude: np.ndarray,
              base_folder: str = None,
              vmax: float = None,
              vmin: float = None,
              save_fig: bool = False,):
    """
    绘制栅格的函数
    :param amplitude: 形如(total_steps, total_components)
    """
    # 创建时间步和频率的网格
    count_steps, count_components = amplitude.shape
    steps = np.arange(count_steps)
    components = np.arange(count_components)

    # 绘制频谱图
    ratio = count_components // count_steps if count_components >= count_steps else 0.5
    plt.figure(figsize=(12, 12*ratio))
    plt.pcolormesh(steps, components, amplitude.T, shading='auto', cmap='bwr',
                   vmax=vmax, vmin=vmin)

    # 添加颜色条
    plt.colorbar(label='Amplitude')

    # 设置坐标轴标签
    plt.xlabel('Variates')
    plt.ylabel('Frequency')
    plt.tight_layout()
    # 显示图形
    if save_fig:
        fig_dir = base_folder
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = os.path.join(fig_dir, f"{title}.png")
        plt.savefig(fig_name)
    else:
        plt.show()
