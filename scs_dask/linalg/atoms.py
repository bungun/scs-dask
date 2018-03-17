import multipledispatch
import functools
import numpy as np
import operator
import dask
import dask
import dask.array as da

import scs_dask.linalg.linear_operator as linop

namespace_atoms = dict()
dispatch = functools.partial(multipledispatch.dispatch, namespace=namespace_atoms)

# @dispatch(da.Array, str, str)
# def graph_dot(array, input_key, output_key, transpose=False, **options):
#     r""" Build dask graph storing as output a linear operator applied to input.

#     Args:
#         array (:obj:`da.Array`): Matrix
#         input_key (:obj:`str`): Key, in some dask graph, of an input
#             vector assumed to be compatibly sized and chunked with
#             the array.
#         output_key (:obj:`str`): Key of an output vector.
#         transpose (:obj:`bool`, optional): If ``True``, form output as
#             :math:`w = A^Tz`; by default form :math:`y = Ax`.

#     Returns:
#         :obj:`dask.sharedict.Sharedict`: dask graph of matrix-vector
#         product assigned to output vector.
#     """
#     if transpose:
#         idx_out, idx_arr, idx_in = 'j', 'ij', 'i'
#         transform = da.transpose
#     else:
#         idx_out, idx_arr, idx_in = 'i', 'ij', 'j'
#         transform = None
#     blks_in = (array.numblocks[1 - int(transpose)],)

#     dsk_out = da.core.top(
#             da.core.dotmany,
#             output_key, idx_out,
#             array.name, idx_arr,
#             input_key, idx_in,
#             leftfunc=transform,
#             numblocks={array.name: array.numblocks, input_key: blks_in})
#     dsk = dask.sharedict.merge(array.dask)
#     dsk.update_with_key(dsk_out, output_key)
#     return dsk

# @dispatch(linop.DLODense, str, str)
# def graph_dot(dense_op, input_key, output_key, transpose=False, **options):
#     """ Implementation of :func:`graph_dot` for a dense linear operator.
#     """
#     return graph_dot(dense_op.data, input_key, output_key, transpose=transpose)

# @dispatch(linop.DLODiagonal, str, str)
# def graph_dot(diag_op, input_key, output_key, **options):
#     """ Implementation of :func:`graph_dot` for a diagonal linear operator.
#     """
#     vec = diag_op.data
#     dsk_out = da.core.top(
#             operator.mul, output_key, 'i', vec.name, 'i', input_key, 'i',
#             numblocks={vec.name: vec.numblocks, input_key: vec.numblocks})
#     dsk = dask.sharedict.merge(diag_op.dask)
#     dsk.update_with_key(dsk_out, output_key)
#     return dsk

# @dispatch(linop.DLOGram, str, str)
# def graph_dot(gram_op, input_key, output_key, **options):
#     """ Implementation of :func:`graph_dot` for a gram operator.
#     """
#     mid_key = gram_op.name + '-gramA-' + input_key
#     dsk_Ax = graph_dot(
#             gram_op.data, input_key, mid_key, transpose=gram_op.transpose)
#     dsk_AAx = graph_dot(
#             gram_op.data, mid_key, output_key, transpose=(not gram_op.transpose))
#     return dask.sharedict.merge(dsk_Ax, dsk_AAx)

# @dispatch(linop.DLORegularizedGram, str, str)
# def graph_dot(gram_op, input_key, output_key, **options):
#     """ Implementation of :func:`graph_dot` for a regularized operator.
#     """
#     mid_key = gram_op.name + '-gramAA-' + input_key
#     blocks = (gram_op.numblocks[0],)
#     def wrap_gram(data):
#         return data if isinstance(data, linop.DLOGram) else linop.DLOGram(data)
#     def add_regularization(AAxi, xi):
#         return AAxi + gram_op.regularization * xi

#     dsk_AAx = graph_dot(wrap_gram(gram_op.data), input_key, mid_key)
#     dsk_IAAx = da.core.top(
#             add_regularization, output_key, 'i', mid_key, 'i', input_key, 'i',
#             numblocks={mid_key: blocks, input_key: blocks})
#     dsk = dask.sharedict.merge(dsk_AAx)
#     dsk.update_with_key(dsk_IAAx, output_key)
#     return dsk


#########

# @dispatch(da.Array, da.Array, da.Array)
# @dispatch(linop.DLODense, da.Array, da.Array)
# @dispatch(linop.DLODense, da.Array, da.Array)
# def graph_dot(array, input_vec, output_vec, transpose=False):
#     r""" Build dask graph storing as output a linear operator applied to input.
#     """
#     return graph_dot(array, input_vec.name, output_vec.name, transpose=transpose)


@dispatch(np.ndarray)
def diag_gram(array, transpose=False, regularization=0., diag_index=0):
    r""" Given matrix :math:`A`, calculate diagonal of :math:`A^TA`.

    Args:
        array (:obj:`np.ndarray`): Dense matrix
        transpose (:obj:`bool`, optional): If ``True``, calculate
            diagonal of :math:`AA^T`
        diag_index (:obj:`int`, optional): Index of subdiagonal to
            calculate.

    Returns:
        :obj:`np.ndarray`: Dense vector representation of requested
        (sub)diagonal of :math`A^TA` (or :math:`AA^T`).

    Raises:
        AssertionError: If diag_index exceeds outer dimension of gram
        matrix :math:`A^TA` (or :math:`AA^T` when ``transpose = True``).
    """
    m, n = array.shape
    dim = m if transpose else n
    diag_index = abs(diag_index)
    assert diag_index < dim, 'subdiagonal index out of range'
    diag = np.zeros(dim - diag_index, dtype=array.dtype)
    if transpose:
        for i in range(dim - diag_index):
            diag[i] = np.dot(array[i + diag_index, :], array[i, :])
    else:
        for i in range(dim - diag_index):
            diag[i] = np.dot(array[:, i + diag_index], array[:, i])

    return diag


# sparse matrices
# other operators?

@dispatch(da.Array)
def diag_gram(array, transpose=False, regularization=0, diag_index=0):
    """ Blocked version of :func:`diag_gram`

    Args:
        array (:obj:`da.Array`): Block-specified matrix
        transpose (:obj:`bool`, optional): If ``True``, calculate
            diagonal of :math:`AA^T`
        diag_index (:obj:`int`, optional): Index of subdiagonal to
            calculate.

    Returns:
        :obj:`da.Array`: Graph-backed dask array representation of
        requested (sub)diagonal of :math`A^TA` (or :math:`AA^T`);
        requires :func:`dask.compute` or :func:`dask.persist` calls
        to convert to be backed by literals.

    Raises:
        AssertionError: If diag_index exceeds outer dimension of gram
        matrix :math:`A^TA` (or :math:`AA^T` when ``transpose = True``).
        NotImplementedError: If diag_index > 0
    """
    # calculate gram chunks from array
    chunks = array.chunks
    nblocks = (len(chunks[0]), len(chunks[1]))
    chunks_gram = (chunks[1 - int(transpose)],)

    # run over gram chunks to figure out diag chunks
    diag_index = abs(int(diag_index))
    m, n = array.shape
    dim = m if transpose else n
    assert diag_index < dim
    if diag_index == 0:
        chunks_calc = chunks_gram
    else:
        raise NotImplementedError
        # TODO: implement for subdiagonals

    # construct graph to populate (sub)diagonal gram chunks
    _diag_ij = 'diag_ij-' + array.name
    _diag_i = 'diag_i-' + array.name
    _diag_i_reg = 'diag_i_reg-' + array.name
    dsk_diag_ij = dict()

    if transpose:
        # row-major algorithm
        # .. for row i
        # .. calculate diag for each block ij in column block i
        # .. sum diagonals over all blocks ij in column block i
        # .. assign to subvector i
        def dg(block, offset=0):
            return diag_gram(block, transpose=True, diag_index=0)
        for i in range(nblocks[0]):
            for j in range(nblocks[1]):
                dsk_diag_ij[(_diag_ij, i, j)] = (dg, (array.name, i, j))
        reduce_diag_j = functools.partial(da.core.reduce, operator.add)
        dsk_diag_i = da.core.top(
                reduce_diag_j, _diag_i, 'i', _diag_ij, 'ij',
                numblocks={_diag_ij: array.numblocks})
    else:
        # column-major algorithm
        # .. for column block j
        # .. calculate diag for each block ij in column block j
        # .. sum diagonals over all blocks ij in column block j
        # .. assign to subvector j
        def dg(block, offset=0):
            return diag_gram(block, transpose=False, diag_index=offset)
        for j in range(nblocks[1]):
            for i in range(nblocks[0]):
                dsk_diag_ij[(_diag_ij, i, j)] = (dg, (array.name, i, j))
        reduce_diag_j = functools.partial(da.core.reduce, operator.add)
        dsk_diag_i = da.core.top(reduce_diag_j, _diag_i, 'i', _diag_ij, 'ji',
                                 numblocks={_diag_ij: array.numblocks})

    if regularization == 0:
        dsk_diag_i_reg = {
                (_diag_i_reg, key[-1]): dsk_diag_i[key]
                for key in dsk_diag_i}
    else:
        def add_regularization(diag): return diag + float(regularization)
        dsk_diag_i_reg = da.core.top(
                add_regularization, _diag_i_reg, 'i', _diag_i, 'i',
                numblocks={_diag_i: (len(chunks_calc[0]),)})

    # retrieve array from graph
    dsk_diag = dask.sharedict.merge(array.dask)
    dsk_diag.update_with_key(dsk_diag_ij, _diag_ij)
    dsk_diag.update_with_key(dsk_diag_i, _diag_i)
    dsk_diag.update_with_key(dsk_diag_i_reg, _diag_i_reg)
    diagg = da.Array(
            dsk_diag, _diag_i_reg, shape=(dim,), chunks=chunks_calc,
            dtype=array.dtype)

    # rechunk
    diagg.rechunk(chunks_gram[0])
    return diagg

@dispatch(linop.DLODense)
def diag_gram(linear_op, transpose=False, regularization=0, diag_index=0):
    return diag_gram(
            linear_op.data, transpose=transpose, regularization=regularization,
            diag_index=diag_index)

@dispatch(linop.DLOGram)
def diag_gram(linear_op, **options):
    return diag_gram(
            linear_op.data, transpose=linear_op.transpose, **options)

@dispatch(linop.DLORegularizedGram)
def diag_gram(linear_op, **options):
    return diag_gram(
            linear_op.data, regularization=linear_op.regularization, **options)


