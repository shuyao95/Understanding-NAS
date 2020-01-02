from collections import namedtuple
import numpy as np
from copy import deepcopy

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1),
            ('sep_conv_5x5', 0), ('skip_connect', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ],
    reduce_concat=[4, 5, 6],
)

NASNet_ADAPT = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1),
            ('sep_conv_5x5', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ],
    normal_concat=[4, 5, 6],
    reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_7x7', 2), ('sep_conv_7x7', 0),
            ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('conv_7x1_1x7', 0), ('sep_conv_3x3', 5), ],
    reduce_concat=[3, 4, 6]
)

AmoebaNet_ADAPT = Genotype(
    normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ],
    normal_concat=[4, 5, 6],
    reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_7x7', 1), ('sep_conv_7x7', 0),
            ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('conv_7x1_1x7', 0), ('sep_conv_3x3', 1), ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

SNAS_MILD = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])

SNAS_ADAPT = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

ENAS = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1),
            ('avg_pool_3x3', 1), ('sep_conv_5x5', 4), ('avg_pool_3x3', 1), ('sep_conv_3x3', 5), ('sep_conv_5x5', 0)],
    reduce_concat=[2, 3, 6])

ENAS_ADAPT = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 0), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
    reduce_concat=[2, 3, 6])


def random_ops(arch, num_ops=8, seed=1):
    np.random.seed(seed)
    normal = np.random.choice(len(PRIMITIVES) - 1, num_ops)
    reduce = np.random.choice(len(PRIMITIVES) - 1, num_ops)
    new_arch = deepcopy(arch)
    for i in range(num_ops):
        new_arch.normal[i] = (PRIMITIVES[1 + normal[i]], arch.normal[i][1])
        new_arch.reduce[i] = (PRIMITIVES[1 + reduce[i]], arch.reduce[i][1])
    return new_arch


# same ops with different connections
def random_conn(arch, num_ops=8, seed=1):
    states = [0, 1]
    new_arch = deepcopy(arch)
    np.random.seed(seed)
    for i in range(num_ops):
        new_arch.normal[i] = (
            arch.normal[i][0], np.random.choice(states, 1).item())
        new_arch.reduce[i] = (
            arch.reduce[i][0], np.random.choice(states, 1).item())
        if i % 2 == 1:
            states.append(states[-1] + 1)
    return new_arch


ENAS_OPS1 = random_ops(ENAS, num_ops=10, seed=1)
ENAS_OPS2 = random_ops(ENAS, num_ops=10, seed=2)
ENAS_OPS3 = random_ops(ENAS, num_ops=10, seed=3)
ENAS_OPS4 = random_ops(ENAS, num_ops=10, seed=4)

ENAS_OPS5 = random_ops(ENAS, num_ops=10, seed=5)
ENAS_OPS6 = random_ops(ENAS, num_ops=10, seed=6)
ENAS_OPS7 = random_ops(ENAS, num_ops=10, seed=7)
ENAS_OPS8 = random_ops(ENAS, num_ops=10, seed=8)
ENAS_OPS9 = random_ops(ENAS, num_ops=10, seed=9)
ENAS_OPS10 = random_ops(ENAS, num_ops=10, seed=10)


ENAS_CONN1 = random_conn(ENAS, num_ops=10, seed=1)
ENAS_CONN2 = random_conn(ENAS, num_ops=10, seed=2)
ENAS_CONN3 = random_conn(ENAS, num_ops=10, seed=3)
ENAS_CONN4 = random_conn(ENAS, num_ops=10, seed=4)

ENAS_CONN5 = random_conn(ENAS, num_ops=10, seed=5)
ENAS_CONN6 = random_conn(ENAS, num_ops=10, seed=6)
ENAS_CONN7 = random_conn(ENAS, num_ops=10, seed=7)
ENAS_CONN8 = random_conn(ENAS, num_ops=10, seed=8)
ENAS_CONN9 = random_conn(ENAS, num_ops=10, seed=9)
ENAS_CONN10 = random_conn(ENAS, num_ops=10, seed=10)


SNAS_OPS1 = random_ops(SNAS_MILD, num_ops=8, seed=1)
SNAS_OPS2 = random_ops(SNAS_MILD, num_ops=8, seed=2)
SNAS_OPS3 = random_ops(SNAS_MILD, num_ops=8, seed=3)
SNAS_OPS4 = random_ops(SNAS_MILD, num_ops=8, seed=4)

SNAS_OPS5 = random_ops(SNAS_MILD, num_ops=8, seed=5)
SNAS_OPS6 = random_ops(SNAS_MILD, num_ops=8, seed=6)
SNAS_OPS7 = random_ops(SNAS_MILD, num_ops=8, seed=7)
SNAS_OPS8 = random_ops(SNAS_MILD, num_ops=8, seed=8)
SNAS_OPS9 = random_ops(SNAS_MILD, num_ops=8, seed=9)
SNAS_OPS10 = random_ops(SNAS_MILD, num_ops=8, seed=10)


SNAS_CONN1 = random_conn(SNAS_MILD, num_ops=8, seed=1)
SNAS_CONN2 = random_conn(SNAS_MILD, num_ops=8, seed=2)
SNAS_CONN3 = random_conn(SNAS_MILD, num_ops=8, seed=3)
SNAS_CONN4 = random_conn(SNAS_MILD, num_ops=8, seed=4)

SNAS_CONN5 = random_conn(SNAS_MILD, num_ops=8, seed=5)
SNAS_CONN6 = random_conn(SNAS_MILD, num_ops=8, seed=6)
SNAS_CONN7 = random_conn(SNAS_MILD, num_ops=8, seed=7)
SNAS_CONN8 = random_conn(SNAS_MILD, num_ops=8, seed=8)
SNAS_CONN9 = random_conn(SNAS_MILD, num_ops=8, seed=9)
SNAS_CONN10 = random_conn(SNAS_MILD, num_ops=8, seed=10)


DARTS_OPS1 = random_ops(DARTS_V2, num_ops=8, seed=1)
DARTS_OPS2 = random_ops(DARTS_V2, num_ops=8, seed=2)
DARTS_OPS3 = random_ops(DARTS_V2, num_ops=8, seed=3)
DARTS_OPS4 = random_ops(DARTS_V2, num_ops=8, seed=4)
DARTS_OPS5 = random_ops(DARTS_V2, num_ops=8, seed=5)

DARTS_OPS6 = random_ops(DARTS_V2, num_ops=8, seed=6)
DARTS_OPS7 = random_ops(DARTS_V2, num_ops=8, seed=7)
DARTS_OPS8 = random_ops(DARTS_V2, num_ops=8, seed=8)
DARTS_OPS9 = random_ops(DARTS_V2, num_ops=8, seed=9)
DARTS_OPS10 = random_ops(DARTS_V2, num_ops=8, seed=10)

DARTS_CONN1 = random_conn(DARTS_V2, num_ops=8, seed=1)
DARTS_CONN2 = random_conn(DARTS_V2, num_ops=8, seed=2)
DARTS_CONN3 = random_conn(DARTS_V2, num_ops=8, seed=3)
DARTS_CONN4 = random_conn(DARTS_V2, num_ops=8, seed=4)
DARTS_CONN5 = random_conn(DARTS_V2, num_ops=8, seed=5)

DARTS_CONN6 = random_conn(DARTS_V2, num_ops=8, seed=6)
DARTS_CONN7 = random_conn(DARTS_V2, num_ops=8, seed=7)
DARTS_CONN8 = random_conn(DARTS_V2, num_ops=8, seed=8)
DARTS_CONN9 = random_conn(DARTS_V2, num_ops=8, seed=9)
DARTS_CONN10 = random_conn(DARTS_V2, num_ops=8, seed=10)


AmoebaNet_OPS1 = random_ops(AmoebaNet, num_ops=10, seed=1)
AmoebaNet_OPS2 = random_ops(AmoebaNet, num_ops=10, seed=2)
AmoebaNet_OPS3 = random_ops(AmoebaNet, num_ops=10, seed=3)
AmoebaNet_OPS4 = random_ops(AmoebaNet, num_ops=10, seed=4)

AmoebaNet_OPS5 = random_ops(AmoebaNet, num_ops=10, seed=5)
AmoebaNet_OPS6 = random_ops(AmoebaNet, num_ops=10, seed=6)
AmoebaNet_OPS7 = random_ops(AmoebaNet, num_ops=10, seed=7)
AmoebaNet_OPS8 = random_ops(AmoebaNet, num_ops=10, seed=8)
AmoebaNet_OPS9 = random_ops(AmoebaNet, num_ops=10, seed=9)
AmoebaNet_OPS10 = random_ops(AmoebaNet, num_ops=10, seed=10)

AmoebaNet_CONN1 = random_conn(AmoebaNet, num_ops=10, seed=1)
AmoebaNet_CONN2 = random_conn(AmoebaNet, num_ops=10, seed=2)
AmoebaNet_CONN3 = random_conn(AmoebaNet, num_ops=10, seed=3)
AmoebaNet_CONN4 = random_conn(AmoebaNet, num_ops=10, seed=4)

AmoebaNet_CONN5 = random_conn(AmoebaNet, num_ops=10, seed=5)
AmoebaNet_CONN6 = random_conn(AmoebaNet, num_ops=10, seed=6)
AmoebaNet_CONN7 = random_conn(AmoebaNet, num_ops=10, seed=7)
AmoebaNet_CONN8 = random_conn(AmoebaNet, num_ops=10, seed=8)
AmoebaNet_CONN9 = random_conn(AmoebaNet, num_ops=10, seed=9)
AmoebaNet_CONN10 = random_conn(AmoebaNet, num_ops=10, seed=10)


NASNet_OPS1 = random_ops(NASNet, num_ops=10, seed=1)
NASNet_OPS2 = random_ops(NASNet, num_ops=10, seed=2)
NASNet_OPS3 = random_ops(NASNet, num_ops=10, seed=3)
NASNet_OPS4 = random_ops(NASNet, num_ops=10, seed=4)

NASNet_OPS5 = random_ops(NASNet, num_ops=10, seed=5)
NASNet_OPS6 = random_ops(NASNet, num_ops=10, seed=6)
NASNet_OPS7 = random_ops(NASNet, num_ops=10, seed=7)
NASNet_OPS8 = random_ops(NASNet, num_ops=10, seed=8)
NASNet_OPS9 = random_ops(NASNet, num_ops=10, seed=9)
NASNet_OPS10 = random_ops(NASNet, num_ops=10, seed=10)

NASNet_CONN1 = random_conn(NASNet, num_ops=10, seed=1)
NASNet_CONN2 = random_conn(NASNet, num_ops=10, seed=2)
NASNet_CONN3 = random_conn(NASNet, num_ops=10, seed=3)
NASNet_CONN4 = random_conn(NASNet, num_ops=10, seed=4)

NASNet_CONN5 = random_conn(NASNet, num_ops=10, seed=5)
NASNet_CONN6 = random_conn(NASNet, num_ops=10, seed=6)
NASNet_CONN7 = random_conn(NASNet, num_ops=10, seed=7)
NASNet_CONN8 = random_conn(NASNet, num_ops=10, seed=8)
NASNet_CONN9 = random_conn(NASNet, num_ops=10, seed=9)
NASNet_CONN10 = random_conn(NASNet, num_ops=10, seed=10)
