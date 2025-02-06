import numpy as np


def read_result_file(filename):
    """
    读取预测文件，输出三个张量inputs, target, prediction，形状都是[I, T, N, C]
    """
    # note 先读取三个文件
    result_obj = np.load(filename)
    target = result_obj['target']
    prediction = result_obj['prediction']
    inputs = result_obj['inputs']
    return inputs, target, prediction
