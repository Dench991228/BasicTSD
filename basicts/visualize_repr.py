import os.path

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
    for f in repr_result_file:
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

