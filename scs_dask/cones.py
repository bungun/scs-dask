import operator
import functools
import numpy as np
import dask
import dask.array as da
import multipledispatch

namespace_cones = dict()
dispatch = functools.partial(multipledispatch.dispatch, namespace=namespace_cones)

class ConvexCone(object):
    def __init__(self, dimension):
        self.dim = int(dimension)

    def __dask_tokenize__(self):
        return (type(self), self.dim)

class ZeroCone(ConvexCone):
    pass

class ZeroDualCone(ConvexCone):
    pass

class NonnegativeCone(ConvexCone):
    def __init__(self, dimension):
        ConvexCone.__init__(self, dimension)

NonnegativeDualCone = NonnegativeCone

class SecondOrderCone(ConvexCone):
    pass

SecondOrderDualCone = SecondOrderCone

class PositiveSemidefiniteCone(ConvexCone):
    pass

PositiveSemidefiniteDualCone = PositiveSemidefiniteCone
PSDCone = PositiveSemidefiniteCone

# exp(x)
# perspective: y*exp(x/y), y > 0
# K = epi[(y*exp(x/y)), y > 0] = {(x, y, z) | y*exp(x/y) >= z, y > 0}
# K_exp = cl(K) = K \cup {(x, 0, z) \in R^3| x =< 0, z >= 0}
class ExponentialCone(ConvexCone):
    pass

# K_exp^* = {(u, v, w) \in R_- \times R \times R_+ | -ulog(-u/w) + u - v <= 0}
#           \cup {(0, v, w) | v >= 0, w >= 0}
class ExponentialDualCone(ConvexCone):
    pass

class PowerCone(ConvexCone):
    def __init__(self, powers):
        powers = np.array(powers)
        assert all(abs(powers) <= 1)
        ConvexCone.__init__(self, len(powers))
        self.powers = powers

class PowerDualCone(PowerCone):
    def __init__(self, powers):
        PowerCone.__init__(self, -1 * np.array(powers))

CONES = (
    'Zero',
    'Nonnegative',
    'SecondOrder',
    'PositiveSemidefinite',
    'Exponential',
    'Power')
DUAL_CONES = {eval(nm + 'Cone'): eval(nm + 'DualCone') for nm in CONES}
DUAL_CONES.update({dc: pc for (pc, dc) in DUAL_CONES.items()})

def K_to_Kstar(K):
    return DUAL_CONES[type(K)](K.dim)

