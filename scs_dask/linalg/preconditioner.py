import operator
import dask
import dask.array as da

from scs_dask.linalg import linear_operator as linop
from scs_dask.linalg import atoms

namespace_linop_preconditioner = dict()
dispatch = functools.partial(
        multipledispatch.dispatch, namespace=namespace_linop_preconditioner)

class DLODiagonalPreconditioner(linop.DLODiagonal):
    def __init__(self, dsk_vector, name=None):
        linop.DLODiagonal.__init__(self, dsk_vector, name=name)




class DLOJacobiPreconditioner(DaskDiagonalPreconditioner):
    def __init__(self, square_linear_operator, regularization=0.):
        diagA = atoms.diag(square_linear_operator)
        rho = float(regularization)
        d = 1. / (diagv(square_linear_operator) + rho)
        DiagonalPreconditioner.__init__(self, d)


@dispatch(linop.DLODiagonal, dict, str, str)
def graph_apply_preconditioner(preconditioner, output_key, input_key):
    diag = preconditioner.array
    dsk_out = da.core.top(
            operator.mul, output_key, 'i', diag.name, 'i', input_key, 'i',
            numblocks={diag.name: diag.numblocks, input_key: diag.numblocks})
    return dask.sharedict.merge(diag.dask, (dsk_out, output_key))
