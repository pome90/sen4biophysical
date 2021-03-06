import numpy as np

try:
    import tensorflow as tf
except ImportError as error:
    print("You must install manualy tensorflow")
    raise error

from sen4biophysical.base import Biophysical

_LAI_NORMLIZIATION = [
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

_LAI_WEIGHTS = [
    [ 
        [-0.023406879, 0.921655165, 0.135576544, -1.938331472, -3.342495816, 0.902277648,0.205363538, -0.040607845, -0.08319641, 0.260029271, 0.284761567],
        [-0.132555481, -0.139574837, -1.014606017, -1.330890039, 0.031730625, -1.433583541, -0.959637899, 1.133115707, 0.216603877, 0.410652304, 0.064760156],
        [0.086015978, 0.616648777, 0.678003876, 0.141102399, -0.096682207, -1.128832639, 0.302189103, 0.434494937, -0.021903699, -0.228492477, -0.039460538],
        [-0.109366594, -0.071046263, 0.064582411, 2.906325237, -0.673873109, -3.838051868, 1.695979345, 0.046950296, -0.049709653, 0.021829545, 0.057483827],
        [-0.089939416, 0.175395483, -0.081847329, 2.219895367, 1.713873975, 0.713069186, 0.138970813, -0.060771762, 0.124263341, 0.21008614, -0.183878139],   
    ],
    [-1.50013549, -0.096283269, -0.194935931, -0.352305896, 0.075107416]
]

_LAI_BIAS = [
    [4.962380306, 1.416008444, 1.075897047, 1.533988265, 3.024115931],
    [1.096963107]
]

_LAI_DENORMALIZATION = [0.000319183, 14.46750945]

_LAI_EXTREME = [-0.2, 0, 8]


class LAI(Biophysical):
    def __init__(self, default_weights=True, **kwargs):
        super(LAI, self).__init__(_LAI_EXTREME, **kwargs)
        if default_weights:
            self.call(tf.zeros([1, 11]))
            self.norm.set_weights([np.transpose(np.array(_LAI_NORMLIZIATION, dtype="float32"), [1, 0])])
            self.d1.set_weights([np.transpose(np.array(_LAI_WEIGHTS[0], dtype="float32"), [1, 0]), np.array(_LAI_BIAS[0], dtype="float32")])
            self.d2.set_weights([np.expand_dims(np.array(_LAI_WEIGHTS[1], dtype="float32"), axis=-1), np.array(_LAI_BIAS[1], dtype="float32")])
            self.denorm.set_weights([np.expand_dims(np.array(_LAI_DENORMALIZATION, dtype="float32"), axis=-1)])
    