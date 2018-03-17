import numpy as np
import dask
import dask.array as da
import scs_dask.linalg.atoms as atoms

# def test_graph_dot_dask_array():
#     A = da.random.random((100, 50), chunks=10)
#     x = da.random.random(50, chunks=10)
#     y = da.random.random(100, chunks=10)

#     dsk_Ax = dask.sharedict.merge(graph_dot(A, x.name, 'Ax'), x.dask)
#     dsk_Ax = dask.sharedict.merge(dsk_Ax, x.dask)
#     Ax = da.Array(
#             dsk_Ax, 'Ax', shape=(A.shape[0],), chunks=(A.chunks[0],),
#             dtype=A.dtype)
#     (diff,) = dask.compute(Ax - A.dot(x))
#     assert np.linalg.norm(diff) < 1e-15

#     dsk_ATy = graph_dot(A, y.name, 'ATy', transpose=True)
#     dsk_ATy = dask.sharedict.merge(dsk_ATy, y.dask)
#     ATy = da.Array(
#             dsk_ATy, 'ATy', shape=(A.shape[1],), chunks=(A.chunks[1],),
#             dtype=A.dtype)
#     (diff, ATy) = dask.compute(ATy - da.transpose(A).dot(y), ATy)
#     assert np.linalg.norm(diff) < 1e-15 * (1 + np.linalg.norm(ATy))

# def test_graph_dot_dense_operator():
#     A = da.random.random((100, 50), chunks=10)
#     x = da.random.random(50, chunks=10)
#     y = da.random.random(100, chunks=10)

#     dsk_Ax = graph_dot(linop.DLODense(A), x.name, 'Ax')
#     dsk_Ax = dask.sharedict.merge(dsk_Ax, x.dask)
#     Ax = da.Array(
#             dsk_Ax, 'Ax', shape=(A.shape[0],), chunks=(A.chunks[0],),
#             dtype=A.dtype)
#     (diff,) = dask.compute(Ax - A.dot(x))
#     assert np.linalg.norm(diff) < 1e-15

#     dsk_ATy = graph_dot(linop.DLODense(A), y.name, 'ATy', transpose=True)
#     dsk_ATy = dask.sharedict.merge(dsk_ATy, y.dask)
#     ATy = da.Array(
#             dsk_ATy, 'ATy', shape=(A.shape[1],), chunks=(A.chunks[1],),
#             dtype=A.dtype)
#     (diff, ATy) = dask.compute(ATy - da.transpose(A).dot(y), ATy)
#     assert np.linalg.norm(diff) < 1e-15 * (1 + np.linalg.norm(ATy))

# def test_graph_dot_diagonal_operator():
#     A = da.random.random((100, 50), chunks=10)
#     x = da.random.random(50, chunks=10)

#     DD = da.random.random(x.shape, chunks=x.chunks)
#     dsk_Dx = graph_dot(linop.DLODiagonal(DD), x.name, 'Dx')
#     dsk_Dx = dask.sharedict.merge(dsk_Dx, x.dask)
#     Dx = da.Array(dsk_Dx, 'Dx', shape=x.shape, chunks=x.chunks, dtype=x.dtype)
#     (diff,) = dask.compute(Dx - DD * x)
#     assert np.linalg.norm(diff) < 1e-15

# def test_graph_dot_gram_operator():
#     A = da.random.random((100, 50), chunks=10)
#     B = da.random.random((50, 100), chunks=10)
#     x = da.random.random(50, chunks=10)

#     # dsk_AAx for A'A
#     dsk_AAx = graph_dot(linop.DLOGram(A, name='ATA'), x.name, 'AAx')
#     dsk_AAx = dask.sharedict.merge(dsk_AAx, x.dask)
#     AAx = da.Array(
#             dsk_AAx, 'AAx', shape=(A.shape[1],), chunks=(A.chunks[1],),
#             dtype=A.dtype)
#     diff = AAx - da.transpose(A).dot(A.dot(x))
#     (diff, AAx) = dask.compute(diff, AAx)
#     assert np.linalg.norm(diff) < 1e-15 * (1 + np.linalg.norm(AAx))

#     # dsk_BBx for BB'
#     dsk_BBx = graph_dot(linop.DLOGram(B, name='BBT'), x.name, 'BBx')
#     dsk_AAx = dask.sharedict.merge(dsk_BBx, x.dask)
#     BBx = da.Array(dsk_BBx, 'BBx', shape=x.shape, chunks=x.chunks, dtype=x.dtype)
#     diff = BBx - B.dot(da.transpose(B).dot(x))
#     (diff, BBx) = dask.compute(diff, BBx)
#     assert np.linalg.norm(diff) < 1e-15 * (1 + np.linalg.norm(BBx))

# def test_graph_dot_reg_gram_operator():
#     A = da.random.random((100, 50), chunks=10)
#     B = da.random.random((50, 100), chunks=10)
#     x = da.random.random(50, chunks=10)
#     mu = 1 + np.random.random()

#     # dsk_IAAx for mu * I + A'A
#     muIAA = linop.DLORegularizedGram(A, regularization=mu)
#     dsk_muIAAx = graph_dot(muIAA, x.name, 'muIAAx')
#     dsk_muIAAx = dask.sharedict.merge(dsk_muIAAx, x.dask)
#     muIAAx = da.Array(
#             dsk_muIAAx, 'muIAAx', shape=(A.shape[1],), chunks=(A.chunks[1],),
#             dtype=A.dtype)
#     diff = muIAAx - (mu * x + da.transpose(A).dot(A.dot(x)))
#     (diff, muIAAx) = dask.compute(diff, muIAAx)
#     assert np.linalg.norm(diff) < 1e-15 * (1 + np.linalg.norm(muIAAx))

#     # dsk_IBBx for mu * I + BB')
#     muIBB = linop.DLORegularizedGram(B, regularization=mu)
#     dsk_muIBBx = graph_dot(muIBB, x.name, 'muIBBx')
#     dsk_muIBBx = dask.sharedict.merge(dsk_muIBBx, x.dask)
#     muIBBx = da.Array(
#             dsk_muIBBx, 'muIBBx', shape=(B.shape[0],), chunks=(B.chunks[0],),
#             dtype=B.dtype)
#     diff = muIBBx - (mu * x + B.dot(da.transpose(B).dot(x)))
#     (diff, muIBBx) = dask.compute(diff, muIBBx)
#     assert np.linalg.norm(diff) < 1e-15 * (1 + np.linalg.norm(muIBBx))

def test_diag_gram_ndarray():
    A = np.random.random((100, 50))
    dATA = atoms.diag_gram(A)
    diff = dATA - np.diag(np.dot(A.T, A))
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dATA))

    dAAT = atoms.diag_gram(A, transpose=True)
    diff = dAAT- np.diag(np.dot(A, A.T))
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dAAT))

def test_diag_gram_dask_array():
    m, n = 1000, 500
    mc, nc = 100, 50
    A = np.random.random((m, n))
    Ad = da.from_array(A, chunks=(mc, nc))

    def dask_matches_numpy(A, transpose=False, regularization=0):
        options = dict(transpose=transpose)
        if regularization > 0:
            options['regularization'] = regularization
        chunks = (nc, mc) if transpose else (mc, nc)
        Ad = da.from_array(A, chunks=chunks)
        diagAA = atoms.diag_gram(A, **options)
        diagAA_dsk = atoms.diag_gram(Ad, **options).compute()
        diff = diagAA - diagAA_dsk
        nrm_diff = np.linalg.norm(diff)
        tol = 1e-14 * (np.log(diff.size) + np.linalg.norm(diagAA))
        return nrm_diff < tol, '{} < {}'.format(nrm_diff, tol)

    reg = np.random.uniform(0.5, 1.5)
    assert dask_matches_numpy(A, transpose=False)
    assert dask_matches_numpy(A, transpose=True)
    assert dask_matches_numpy(A, transpose=False, regularization=reg)
    assert dask_matches_numpy(A, transpose=True, regularization=reg)
