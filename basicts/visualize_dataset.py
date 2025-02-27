import numpy as np

from basicts.utils.result import read_result_file
from basicts.utils.visualize import plot_line

def timeseries_visualize(result_file_dir: str, window_index: int, sensor_id: int=0):
    """
    时间序列展示，展示sensor_id的测试集时间序列变化，并且在window_index处展示一条直线
    """
    inputs, targets, predictions = read_result_file(result_file_dir)
    input_windows = inputs[:, :, sensor_id, 0]
    x = np.arange(inputs.shape[0])
    # 绘图
    plot_line(x=x, ys=[input_windows[:, 0]], y_names=["input series"], y_axis_name="y", w=window_index, title=f"Shape of input series")

def window_similarity_visualize(result_file_dir: str, window_index: int, sensor_id: int = 0):
    """
    计算一个特定的输入窗口与其它输入窗口之间的输入，输出之间的相似性，用来衡量node层面的时序不可分辨性
    :param result_file_dir: 结果文件的.npz的位置
    :param window_index: 窗口的index
    """
    # (I, Timestamps, Nodes, Feature)
    inputs, targets, predictions = read_result_file(result_file_dir)
    # 各个需要被展示的输入、输出窗口 (I, T)
    input_windows = inputs[:, :, sensor_id, 0]
    target_windows = targets[:, :, sensor_id, 0]
    # 需要对比的输入和输出窗口 (1, T)
    referred_input_window = np.expand_dims(input_windows[window_index], 0)
    referred_output_window = np.expand_dims(target_windows[window_index], 0)

    # 计算各个输入窗口的输入部分与参考输入窗口的mse以及输出部分与参考输出窗口的mse
    # (N, T) -> (N)
    referred_input_mse = np.mean((referred_input_window - input_windows)**2,  axis=-1)
    referred_output_mse = np.mean((referred_output_window - target_windows)**2, axis=-1)
    x = np.arange(inputs.shape[0])

    plot_line(x=x, ys=[referred_input_mse, referred_output_mse], y_names=["input_window_distance", "output_window_distance"], y_axis_name="MSE Distance", w=window_index, title=f"Similarity of input and output windows referring to window {window_index} of sensor {sensor_id}")
