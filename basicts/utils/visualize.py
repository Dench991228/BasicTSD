import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt


def plot_line(base_folder: str, title: str,
              x: List|np.ndarray, ys: List, y_names: List,
              y_max = None, y_min = None,
              x_axis_name: str = "time steps", y_axis_name: str = "loss"
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
    """
    # 创建一个新的图形
    plt.figure(figsize=(24, 6))

    # 挨个绘制折线
    for y, y_name in zip(ys, y_names):
        plt.plot(x, y, label=y_name, marker='.', linestyle='-')

    # 添加标题
    plt.title(title)

    # 添加X轴和Y轴标签
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.ylim(y_min, y_max)
    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)
    plt.tight_layout()
    # 显示图形
    fig_name = os.path.join(base_folder, f"{title}.png")
    plt.savefig(fig_name)


def plot_mesh():
    """
    绘制栅格的函数
    """
    pass