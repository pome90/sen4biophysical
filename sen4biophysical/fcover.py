import numpy as np

try:
    import tensorflow as tf
except ImportError as error:
    print("You must install manualy tensorflow")
    raise error

from sen4biophysical.base import Biophysical

_FCOVER_NORMLIZIATION = [
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

_FCOVER_WEIGHTS = [
    [ 
        [-0.156854265, 0.124234528, 0.235625516, -1.832391026, -0.21718897, 5.069339581, -0.887578008, -1.080846817, -0.032316704, -0.224476137, -0.195523963],
        [-0.220824928, 1.285953955, 0.703139486, -1.344812167, -1.968812676, -1.454446816, 1.0273756, -0.124946415, 0.080276244, -0.198705919, 0.108527101],
        [-0.409688743, 1.088588848, 0.362845226, 0.036939051, -0.34801259, -2.003526188, 0.04103576, 1.223738532, -0.012408278, -0.282223365, 0.099499312],
        [-0.188970958, -0.035862184, 0.005512485, 1.353915708, -0.739689896, -2.217195301, 0.313216124, 1.502016891, 1.215304902, -0.421938359, 1.488524845],
        [2.492939937, -4.405113314, -1.910620126, -0.703174116, -0.215104721, -0.972151495, -0.930752241, 1.214344188, -0.52166546, -0.445755956, 0.344111874]
    ],
    [0.230805868, -0.333655485, -0.499418292, 0.04724844, -0.079851654]
]

_FCOVER_BIAS = [
    [-1.452616522, -1.704174776, 1.021689658, -0.49800281, -3.889221548],
    [-0.096799815]
]

_FCOVER_DENORMALIZATION = [0.000181231, 0.999638215]

_FCOVER_EXTREME = [-0.1, 0, 1]


class FCOVER(Biophysical):
    def __init__(self, default_weights=True, **kwargs):
        super(FCOVER, self).__init__(_FCOVER_EXTREME, **kwargs)
        if default_weights:
            self.call(tf.zeros([1, 11]))
            self.norm.set_weights([np.transpose(np.array(_FCOVER_NORMLIZIATION, dtype="float32"), [1, 0])])
            self.d1.set_weights([np.transpose(np.array(_FCOVER_WEIGHTS[0], dtype="float32"), [1, 0]), np.array(_FCOVER_BIAS[0], dtype="float32")])
            self.d2.set_weights([np.expand_dims(np.array(_FCOVER_WEIGHTS[1], dtype="float32"), axis=-1), np.array(_FCOVER_BIAS[1], dtype="float32")])
            self.denorm.set_weights([np.expand_dims(np.array(_FCOVER_DENORMALIZATION, dtype="float32"), axis=-1)])      
        