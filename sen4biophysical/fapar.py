import tensorflow as tf
import numpy as np

from sen4biophysical.base import Biophysical

_FAPAR_NORMLIZIATION = [
    [0, 0.25306152],
    [0, 0.290393578],
    [0, 0.305398915],
    [0.006637973, 0.608900396],
    [0.013972727, 0.753827384],
    [0.026690138, 0.782011771],
    [0.016388074, 0.493761398],
    [0, 0.493025984],
    [0.918595401, 1],
    [0.342022871, 0.936206429],
    [-0.999999982, 0.999999999]
]

_FAPAR_WEIGHTS = [
    [ 
        [0.268714455, -0.205473108, 0.281765694, 1.337443412, 0.390319213, -3.612714342, 0.222530961, 0.82179055, -0.093664567, 0.019290146, 0.037364446],
        [-0.248998055, -0.571461305, -0.369957603, 0.246031695, 0.332536215, 0.438269896, 0.819000552, -0.934931499, 0.082716248, -0.286978634, -0.035890968],
        [-0.164063575, -0.126303286, -0.253670784, -0.321162835, 0.067082288, 2.029832289, -0.023141229, -0.553176626, 0.059285452, -0.034334455, -0.031776704],
        [0.130240753, 0.236781036, 0.131811664, -0.250181799, -0.01136415, -1.857573215, -0.146860751, 0.528008831, -0.046230769, -0.034509608, 0.031884395],
        [-0.029929946, 0.795804414, 0.348025318, 0.943567008, -0.27634167, -2.94659418, 0.289483074, 1.04400695, -0.000413032, 0.403331115, 0.068427131]
    ],
    [2.126038811, -0.632044933, 5.598995787, 1.770444141, -0.267879584]
]

_FAPAR_BIAS = [
    [-0.887068364, 0.320126471, 0.610523703, -0.379156191, 1.35302339669],
    [-0.336431284]
]

_FAPAR_DENORMALIZATION = [0.000153013, 0.977135097]

_FAPAR_EXTREME = [-0.1, 0, 0.94]

  
class FAPAR(Biophysical):
    def __init__(self, default_weights=True, **kwargs):
        super(FAPAR, self).__init__(_FAPAR_EXTREME, **kwargs)
        if default_weights:
            self.call(tf.zeros([1, 11]))
            self.norm.set_weights([np.transpose(np.array(_FAPAR_NORMLIZIATION, dtype="float32"), [1, 0])])
            self.d1.set_weights([np.transpose(np.array(_FAPAR_WEIGHTS[0], dtype="float32"), [1, 0]), np.array(_FAPAR_BIAS[0], dtype="float32")])
            self.d2.set_weights([np.expand_dims(np.array(_FAPAR_WEIGHTS[1], dtype="float32"), axis=-1), np.array(_FAPAR_BIAS[1], dtype="float32")])
            self.denorm.set_weights([np.expand_dims(np.array(_FAPAR_DENORMALIZATION, dtype="float32"), axis=-1)])
        