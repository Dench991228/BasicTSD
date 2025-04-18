import torch


def load_parameter(parameter_path: str):
    """
    这个函数只有一个目的，那就是加载模型的参数
    """
    return torch.load(parameter_path)['model_state_dict']