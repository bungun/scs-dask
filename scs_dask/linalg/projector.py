import functools

import dask
import dask.array as da
# from scs_dask.linalg import conjugate_gradient as cg

from scs_dask.linalg import linear_operator as linop
from scs_dask.linalg import atoms2
from scs_dask.linalg import cg_variants as cg

def dask_to_array(build_array, key, *dsks):
    build_array(dask.sharedict.merge(*dsks), key)

def cgls_project(A, x, y, tol=1e-8, **options):
    r""" Project (x, y) onto graph G = {(y, x) | y = Ax} via CGLS

    In particular, form outputs as:

        :math:`x_{out} = x + argmin_x 1/2 \|Ax' - (y - Ax)\| + 1/2 \|x'\|_2^2`
        :math:`y_{out} = Ax_{out}`
    """
    fmt = 'array {} compatible'
    assert A.shape[0] == y.shape[0] and A.shape[1] == x.shape[0], fmt.format('dims')
    assert A.chunks[0] == y.chunks[0] and A.chunks[1] == x.chunks[0], fmt.format('chunks')

    token = options.pop('name', 'cgls-project-' + dask.base.tokenize(A, x, y, tol, **options))
    nm_b, nm_x, nm_y = map(lambda nm: nm + '-' + token, ('b', 'x', 'y'))

    # b = y - Ax
    # x_cg = argmin \|Ax' - (b)\| + \|x'\|_2^2
    b = atoms2.gemv(-1, A, x, 1, y, name=nm_b)
    x_cg, res, iters = cg.cgls(A, b, 1, tol=tol)
    x_out = da.add(x, x_cg, name=nm_x)
    y_out = atoms2.dot(A, x_out, name=nm_y)
    return x_out, y_out, res, iters

def cg_project(A, x, y, tol=1e-8, **options):
    r""" Project (x, y) onto graph G = {(y, x) | y = Ax} via CG

    In particular, form outputs as:

        :math:`x_{out} = (1 + A^TA)^{-1}(A^Ty + x)`
        :math:`y_{out} = Ax_{out}`
    """
    fmt = 'array {} compatible'
    assert A.shape[0] == y.shape[0] and A.shape[1] == x.shape[0], fmt.format('dims')
    assert A.chunks[0] == y.chunks[0] and A.chunks[1] == x.chunks[0], fmt.format('chunks')

    token = options.pop('name', 'cg-project-' + dask.base.tokenize(A, x, y, tol, **options))
    nm_b, nm_x, nm_y = map(lambda nm: nm + '-' + token, ('b', 'x', 'y'))

    # b = A'y + x
    b = atoms2.gemv(1, A, y, 1, x, transpose=True, name=nm_b)
    A_hat = linop.DLORegularizedGram(A, transpose=False)
    x_out, res, iters = cg.cg_graph(A_hat, b, tol=tol, name=nm_x, **options)
    y_out = atoms2.dot(A, x_out, name=nm_y)
    x_out, y_out = dask.persist(x_out, y_out)
    return x_out, y_out, res, iters

