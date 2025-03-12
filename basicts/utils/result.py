import os

import numpy as np


def read_result_file(filename):
    """
    读取预测文件，输出三个张量inputs, target, prediction，形状都是[I, T, N, C]
    """
    # note 先读取三个文件
    result_obj = np.load(filename)
    return result_obj

def read_repr_file(filename):
    """
    读取预测文件，输出三个张量inputs, target, prediction，形状都是[I, N, feature]
    """
    # note 先读取三个文件
    result_obj = np.load(filename)
    repr = result_obj['reprs']
    return repr

def data_group_by_node(filename: str):
    """
    读取预测文件，输出一个张量，形状为(node, I, C)，其中node为节点数目，I为测试集有多少时间步，C为变量数目
    """
    # note 整理mae随时间变化
    # 形状为(I, T, N, C)
    output_items = read_result_file(filename)
    inputs = output_items['inputs']
    target = output_items['target']
    prediction = output_items['prediction']
    # (I, T, N, C)
    error = np.abs(target-prediction)
    # (I, T, N, C) -> (I, N, C)
    mae = np.mean(error, axis=1)
    # (I, N, C) -> (N, I, C)
    mae = np.transpose(mae, (1, 0, 2))
    # note 整理输入值随时间变化
    return mae, inputs[:, -1, :, :].transpose((1, 0, 2))

def data_group_by_node_and_cycle(filename: str, cycle: int):
    """
    读取预测的文件，输出一个张量，形状为(node, cycle, C)，cycle是循环一圈的时间步
    """
    # (N, I, C)
    # note 先规整一下mae的循环
    mae_by_node, inputs = data_group_by_node(filename)
    sum_cycles = mae_by_node.shape[1] // cycle
    remaining_timestamps = sum_cycles * cycle
    mae_by_node = mae_by_node[:, :remaining_timestamps]
    # (N, count_cycles, cycle, C) -> (N, cycle, C)
    mae_by_node = mae_by_node.reshape(mae_by_node.shape[0], sum_cycles, cycle, mae_by_node.shape[-1])
    new_shape = mae_by_node.shape
    mae_by_node = np.mean(mae_by_node, axis=1)
    # note 再规整一下原始数据的季节循环
    inputs = inputs[:, :remaining_timestamps]
    inputs = inputs.reshape(new_shape)
    inputs = np.mean(inputs, axis=1)
    return mae_by_node, inputs


def stat_node(file_dir: str, shape):
    """
    展示一个数据集，每一个节点的度数之和以及每一个节点的总交通流量，按照连接数量从大到小输出
    """
    # note 先展示哪一些节点连接量比较大
    adj_path = os.path.join(file_dir, "adj_mx.pkl")
    adj_mx = np.load(adj_path, allow_pickle=True)
    out_degree = np.sum(adj_mx, axis=1)
    in_degree = np.sum(adj_mx, axis=0)
    degree = in_degree + out_degree
    index_degree = np.argsort(degree, axis=0, )
    #print(index_degree[::-1])
    #print(degree[index_degree[::-1]])
    # note 再展示哪一个节点流量比较大
    # (I, N)
    data = np.fromfile(os.path.join(file_dir, "data.dat"), dtype=np.float32).reshape(shape)[:, :, 0]
    data = np.sum(data, axis=0)
    index_flow = np.argsort(data, axis=0,)[::-1]
    #print(index_flow)
    #print(data[index_flow])
    return index_degree[::-1], degree[index_degree[::-1]], index_flow, data[index_flow]
