import numpy as np

from sen4biophysical.base import Biophysical

try:
    import tensorflow as tf
except ImportError as error:
    print("You must install manualy tensorflow")
    raise error

_CW_NORMLIZIATION = [
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

_CW_WEIGHTS = [
    [ 
        [0.14637871, 1.189799282, -0.90623514, -0.808337509, -0.973334918, -1.425912776, -0.005612536, -0.634520356, -0.11722606, -0.060270091, 0.229407587],
        [0.283319173, 0.149342023, 1.084805884, -0.138658791, -0.455759407, 0.420571438, -1.737294904, -0.704286287, 0.019095378, -0.039397132, -0.007502416],    
        [-0.197487428, -0.105460326, 0.158347671, 2.149124267, -0.970716843, -4.927253179, 1.420343018, 1.453169172, 0.022725705, 0.26929865, 0.084904766],
        [0.1414058, 0.333862603, 0.356218929, -0.545942268, 0.089104308, 0.919298363, -1.852089263, -0.427539591, 0.007913856, 0.01483332, -0.001537868],
        [-0.186781083, -0.549163705, -0.181287639, 0.968640437, -0.470442559, -1.248597252, 2.670149423, 0.490090624, -0.001449319, 0.003148294, 0.020651788]
    ],
    [-0.077555589, -0.864117861, -0.199212415, 1.987304612, 0.458926743]
]

_CW_BIAS = [
    [-2.106408369, -1.690220948, 3.101176553, -1.312316265, 1.011319303],
    [-0.19759171]
]

_CW_DENORMALIZATION = [3.85067e-06, 0.522417055]

_CW_EXTREME = [-0.015, 0, 0.55]


class CW(Biophysical):
    def __init__(self, default_weights=True, **kwargs):
        super(CW, self).__init__(_CW_EXTREME, **kwargs)
        if default_weights:
            self.call(tf.zeros([1, 11]))
            self.norm.set_weights([np.transpose(np.array(_CW_NORMLIZIATION, dtype="float32"), [1, 0])])
            self.d1.set_weights([np.transpose(np.array(_CW_WEIGHTS[0], dtype="float32"), [1, 0]), np.array(_CW_BIAS[0], dtype="float32")])
            self.d2.set_weights([np.expand_dims(np.array(_CW_WEIGHTS[1], dtype="float32"), axis=-1), np.array(_CW_BIAS[1], dtype="float32")])
            self.denorm.set_weights([np.expand_dims(np.array(_CW_DENORMALIZATION, dtype="float32"), axis=-1)])
    