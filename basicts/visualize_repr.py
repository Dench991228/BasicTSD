import os.path
from typing import List

import numpy as np

from basicts.utils.result import read_repr_file
from basicts.utils.visualize import  plot_line


def visualize_repr(repr_result_file, window_id: int, sensor_id: int):
    """
    展现一个模型编码的特征与这个节点其它时刻的特征的相似度
    """
    # (I, Node, Feature) -> (I, Features)
    cos = []
    x = None
    for idx, f in enumerate(repr_result_file):
        reprs = read_repr_file(f)
        sensor_features = reprs[:, sensor_id, :]
        sensor_feature_norms = np.linalg.norm(sensor_features, axis=1, keepdims=True)
        sensor_features_normalized = sensor_features / sensor_feature_norms
        # (1, Features)
        referred_sensor_features = np.expand_dims(sensor_features[window_id, :], axis=0)
        referred_sensor_feature_norms = np.linalg.norm(referred_sensor_features, axis=1, keepdims=True)
        referred_sensor_features_normalized = referred_sensor_features / referred_sensor_feature_norms
        # (1, Features) * (Features, I)
        cosine_distances = 1 - np.matmul(referred_sensor_features_normalized, sensor_features_normalized.T)[0]
        cos.append(cosine_distances)
        x = np.arange(reprs.shape[0])
    plot_line(title=f"Feature Similarity Referring to window {window_id} of sensor {sensor_id}",
              x=x, ys=cos, y_names=["feature distance" for i in range(len(cos))], y_axis_name="cosine_distance", w=window_id,
              base_folder=os.path.dirname(repr_result_file[0]))

def compute_cosine_similarity(original_features: np.ndarray, window_id: int):
    """
    计算前面所有特征与后面的特征的相似度
    :param original_features: (I, F)

    """
    # note 先计算前后两者的标准化
    # (I, F)
    norm_features = np.linalg.norm(original_features, axis=1, keepdims=True)
    normalized_features = original_features / norm_features
    # (1, F)
    referred_feature = np.expand_dims(normalized_features[window_id, :], axis=0)
    cosine_distance = 1 - np.matmul(referred_feature, normalized_features.T)[0]
    return cosine_distance

def visualize_repr_decompose(repr_result_file, window_ids: List[int], sensor_ids: List[int]):
    """
    展现一个模型编码的特征与这个节点其它时刻的特征的相似度
    """
    # (I, Node, Feature) -> (I, Features)
    reprs = read_repr_file(repr_result_file)
    total_preds, sensors, feature_dim = reprs.shape
    for sensor_id, window_id in zip(sensor_ids, window_ids):
        cos = []
        feature = reprs[:, sensor_id, :].reshape(total_preds, 12, -1)
        print(feature.shape)
        feature_dim_per_step = feature.shape[-1]
        # 每一步前半部分是trend，后半部分是seasonal
        sensor_features_trend = feature[:,  :, :feature_dim_per_step//2]
        print(sensor_features_trend.shape)
        sensor_features_trend = sensor_features_trend.reshape(total_preds, -1)

        sensor_features_seasonal = feature[:, :,  feature_dim_per_step//2:]
        print(sensor_features_seasonal.shape)
        sensor_features_seasonal = sensor_features_seasonal.reshape(total_preds, -1)
        cos.append(compute_cosine_similarity(sensor_features_trend, window_id))
        cos.append(compute_cosine_similarity(sensor_features_seasonal, window_id))
        x = np.arange(reprs.shape[0])
        plot_line(title=f"Feature Similarity Referring to window {window_id} of sensor {sensor_id}",
                  x=x, ys=cos, y_names=[f"feature distance {s}" for s in ['trend', 'seasonal']], y_axis_name="cosine_distance", w=window_id,
                  base_folder=os.path.dirname(repr_result_file))
