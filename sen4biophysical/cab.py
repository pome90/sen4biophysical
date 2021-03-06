import numpy as np

try:
    import tensorflow as tf
except ImportError as error:
    print("You must install manualy tensorflow")
    raise error

from sen4biophysical.base import Biophysical

_CAB_NORMLIZIATION = [
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

_CAB_WEIGHTS = [
    [ 
        [0.400396555, 0.607936279, 0.137468651, -2.955866573, -3.186746688, 2.206800751, -0.313784336, 0.256063548, -0.07161322, 0.510113504, 0.142813982],
        [-0.250781102, 0.439086303, -1.160590938, -1.86193525, 0.981359868, 1.634230834, -0.872527935, 0.448240475, 0.037078084, 0.03004419, 0.005956687],
        [0.552080133, -0.502919673, 6.105041925, -1.294386119, -1.059956388, -1.394092902, 0.324752733, -1.758871823, -0.03666368, -0.183105291, -0.038145312],
        [0.211591185, -0.248788896, 0.887151598, 1.143675896, -0.75396883, -1.185456953, 0.54189786, -0.252685835, -0.023414901, -0.046022504, -0.006570284],
        [0.254790234, -0.724968611, 0.731872806, 2.303453821, -0.849907967, -6.425315501, 2.238844558, -0.199937574, 0.097303332, 0.334528255, 0.113075307]       
    ],
    [-0.352760041, -0.603407399, 0.135099379, -1.735673124, -0.147546813]
]

_CAB_BIAS = [
    [4.24229967, -0.259569088, 3.130392627, 0.774423577, 2.584276649],
    [0.463426464]
]

_CAB_DENORMALIZATION = [0.007426693, 873.9082221]

_CAB_EXTREME = [-15, 0, 600]

class CAB(Biophysical):
    def __init__(self, default_weights=True, **kwargs):
        super(CAB, self).__init__(_CAB_EXTREME, **kwargs)
        if default_weights:
            self.call(tf.zeros([1, 11]))
            self.norm.set_weights([np.transpose(np.array(_CAB_NORMLIZIATION, dtype="float32"), [1, 0])])
            self.d1.set_weights([np.transpose(np.array(_CAB_WEIGHTS[0], dtype="float32"), [1, 0]), np.array(_CAB_BIAS[0], dtype="float32")])
            self.d2.set_weights([np.expand_dims(np.array(_CAB_WEIGHTS[1], dtype="float32"), axis=-1), np.array(_CAB_BIAS[1], dtype="float32")])
            self.denorm.set_weights([np.expand_dims(np.array(_CAB_DENORMALIZATION, dtype="float32"), axis=-1)])