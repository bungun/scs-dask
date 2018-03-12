import operator
import functools
import multipledispatch
import numpy as np
import dask
import dask.array as da

from scs_dask.linalg import linear_operator as linop
from scs_dask.linalg import atoms

namespace_preconditioner = dict()
dispatch = functools.partial(
        multipledispatch.dispatch, namespace=namespace_preconditioner)

@dispatch(np.ndarray)
def jacobi_preconditioner(array):
    return np.diag(1. / np.diag(array))

@dispatch(da.Array)
def jacobi_preconditioner(array, name=None):
    name = 'jacobi-precond-' + array.name if name is None else name
    m, n = array.shape
    assert m == n, 'preconditioner expects square linear operator'
    diag = da.diag(array)
    return linop.DLODiagonal(da.core.map_blocks(da.reciprocal, diag, name=name))

@dispatch(linop.DLODense)
def jacobi_preconditioner(linear_op, name=None):
    return jacobi_preconditioner(linear_op.data, name=name)

@dispatch(linop.DLOGram)
def jacobi_preconditioner(linear_op, name=None):
    name = 'jacobi-precond-' + linear_op.name if name is None else name
    diag = atoms.diag_gram(linear_op)
    return linop.DLODiagonal(da.core.map_blocks(da.reciprocal, diag, name=name))