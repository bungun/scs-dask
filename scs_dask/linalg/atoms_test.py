import numpy as np
import dask
import dask.array as da
import scs_dask.linalg.atoms as atoms

def test_diag_gram_ndarray():
    A = np.random.random((100, 50))
    dATA = atoms.diag_gram(A)
    diff = dATA - np.diag(np.dot(A.T, A))
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dATA))

    dAAT = atoms.diag_gram(A, tranpose=True)
    diff = dAAT- np.diag(np.dot(A, A.T))
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dAAT))

def test_diag_gram_dask_array():
    m, n = 1000, 500
    mc, nc = 100, 50
    A = dask.random.random(shape=(m, n), chunks=(mc, nc))

    dATA = atoms.diag_gram(A)
    diff = dATA - da.diag(da.dot(A.T, A))
    diff, dATA = dask.compute(diff, dATA)
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dATA))

    dAAT = atoms.diag_gram(A, tranpose=True)
    diff = dAAT - da.diag(da.dot(A, A.T))
    diff, dAAT = dask.compute(diff, dAAT)
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dAAT))

    dATA = diag_gram(A, regularization=1)
    diff = dATA - (1 + da.diag(da.dot(A.T, A)))
    diff, dATA = dask.compute(diff, dATA)
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dATA))

    dATA = diag_gram(A, regularization=1)
    diff = dATA - (1 + da.diag(da.dot(A.T, A)))
    diff, dATA = dask.compute(diff, dATA)
    logdim = np.log(diff.size)
    assert np.linalg.norm(diff) < 1e-15 * (logdim + np.linalg.norm(dATA))