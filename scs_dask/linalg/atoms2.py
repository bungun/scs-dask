import multipledispatch
import functools
import numpy as np
import operator
import dask
import dask
import dask.array as da

import scs_dask.linalg.linear_operator as linop

namespace_atoms2 = dict()
dispatch = functools.partial(multipledispatch.dispatch, namespace=namespace_atoms2)

def graph_gemv(key_alpha, A, key_x, key_beta, key_y, key_out, transpose=False):
    """ For A <: dask.array.Array """
    vblocks, hblocks = A.numblocks
    key_Ax = A.name + '-mul-' + key_x
    dsk = dict()
    def gemv(alpha, Axi, beta, yi): return alpha * Axi + beta * yi
    for i in range(A.numblocks[int(transpose)]):
        dsk[(key_out, i)] = (gemv, key_alpha, (key_Ax, i), key_beta, (key_y, i))
    dsk.update(graph_dot(A, key_x, key_Ax, transpose=transpose))
    return dsk

def gemv(alpha, A, x, beta, y, transpose=False, name=None):
    token = name if name is not None else (
            'gemv-' + dask.base.tokenize(alpha, A, x, beta, y, transpose))
    dsks = [A.dask, x.dask, y.dask]

    def get_key(scalar):
        return (scalar.name,) if scalar.shape == () else (scalar.name, 0)

    if isinstance(alpha, da.Array):
        dsks += [alpha.dask]
        alpha = get_key(alpha)
    if isinstance(beta, da.Array):
        dsks += [beta.dask]
        beta = get_key(beta)

    dsks += [graph_gemv(alpha, A, x.name, beta, y.name, token, transpose=transpose)]
    dsk = dask.sharedict.merge(*dsks)
    return da.Array(dsk, token, shape=y.shape, chunks=y.chunks, dtype=y.dtype)

graph_gemvT = functools.partial(graph_gemv, transpose=True)
gemvT = functools.partial(gemv, transpose=True)

def dot(A, x, transpose=False, name=None, **options):
    token = name if name is not None else 'dot-' + dask.base.tokenize(A, x, transpose)
    dsks = [A.dask, x.dask, graph_dot(A, x.name, token, transpose=transpose, **options)]
    return da.Array(
            dask.sharedict.merge(*dsks),
            token,
            shape=(A.shape[int(transpose)],),
            chunks=(A.chunks[int(transpose)],),
            dtype=A.dtype)

@dispatch(da.Array, str, str)
def graph_dot(array, input_key, output_key, transpose=False, **options):
    """ TODO: docstring """
    matvec = functools.partial(da.core.dotmany, leftfunc=np.transpose) if transpose else da.core.dotmany
    def Aij(i, j): return (array.name, j, i) if transpose else (array.name, i, j)
    blocks_out, blocks_in = array.numblocks[::-1] if transpose else array.numblocks
    dsk = dict()
    for i in range(blocks_out):
        dsk[(output_key, i)] = (matvec,
                             [Aij(i, j) for j in range(blocks_in)],
                             [(input_key, j) for j in range(blocks_in)])
    return dsk

@dispatch(linop.DLODense, str, str)
def graph_dot(dense_op, input_key, output_key, transpose=False, **options):
    """ Implementation of :func:`graph_dot` for a dense linear operator.
    """
    return graph_dot(dense_op.data, input_key, output_key, transpose=transpose)

@dispatch(linop.DLODiagonal, str, str)
def graph_dot(diag_op, input_key, output_key, **options):
    """ Implementation of :func:`graph_dot` for a diagonal linear operator.
    """
    vec = diag_op.data
    dsk = dict()
    for i in range(vec.numblocks[0]):
        dsk[(output_key, i)] = (operator.mul, (vec.name, i), (input_key, i))
    return dsk

@dispatch(linop.DLOGram, str, str)
def graph_dot(gram_op, input_key, output_key, **options):
    """ Implementation of :func:`graph_dot` for a gram operator.
    """
    mid_key = gram_op.name + '-gramA-' + input_key
    dsk = graph_dot(gram_op.data, input_key, mid_key, transpose=gram_op.transpose)
    dsk.update(graph_dot(gram_op.data, mid_key, output_key, transpose=(not gram_op.transpose)))
    return dsk

@dispatch(linop.DLORegularizedGram, str, str)
def graph_dot(gram_op, input_key, output_key, **options):
    """ Implementation of :func:`graph_dot` for a regularized operator.
    """
    mid_key = gram_op.name + '-gramAA-' + input_key
    def wrap_gram(data):
        return data if isinstance(data, linop.DLOGram) else linop.DLOGram(data, transpose=gram_op.transpose)
    def add_regularization(AAxi, xi):
        return AAxi + gram_op.regularization * xi

    dsk = graph_dot(wrap_gram(gram_op.data), input_key, mid_key)
    for i in range(gram_op.numblocks[0]):
        dsk[(output_key, i)] = (add_regularization, (mid_key, i), (input_key, i))
    return dsk