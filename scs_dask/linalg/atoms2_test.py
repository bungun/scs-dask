import functools
import numpy as np
import dask
import dask.array as da

from scs_dask.linalg import linear_operator as linop
from scs_dask.linalg import atoms2

import unittest

def graph_dot_matches_expected(matrixlike, vec, expected, transpose=False):
    tol = 1e-15 * (1 + expected.size**0.5)
    dsk_output = atoms2.graph_dot(matrixlike, vec.name, 'output', transpose=transpose)
    output = da.Array(
            dask.sharedict.merge(matrixlike.dask, vec.dask, dsk_output),
            'output',
            shape=expected.shape,
            chunks=expected.chunks,
            dtype=expected.dtype)
    norm = da.linalg.norm(output - expected).compute()
    assert norm < tol, '|Ax - Ax_expected| < tol: {:.2e} < {:.2e}'.format(norm, tol)
    return True

def dot_matches_expected(matrixlike, vec, expected, transpose=False):
    tol = 1e-15 * (1 + expected.size**0.5)
    output = atoms2.dot(matrixlike, vec, transpose=transpose)
    norm = da.linalg.norm(output - expected).compute()
    assert norm < tol, '|Ax - Ax_expected| < tol: {:.2e} < {:.2e}'.format(norm, tol)
    return True

def graph_gemv_matches_expected(alpha, matrixlike, x, beta, y, expected, transpose=False):
    tol = 1e-15 * (1 + expected.size**0.5)
    dsk_output = atoms2.graph_gemv(
            alpha, matrixlike, x.name, beta, y.name, 'output',
            transpose=transpose)
    output = da.Array(
            dask.sharedict.merge(matrixlike.dask, x.dask, y.dask, dsk_output),
            'output',
            shape=expected.shape,
            chunks=expected.chunks,
            dtype=expected.dtype)
    norm = da.linalg.norm(output - expected).compute()
    assert norm < tol, '|Ax - Ax_expected| < tol: {:.2e} < {:.2e}'.format(norm, tol)
    return True

def gemv_matches_expected(alpha, matrixlike, x, beta, y, expected, transpose=False):
    tol = 1e-15 * (1 + expected.size**0.5)
    output = atoms2.gemv(alpha, matrixlike, x, beta, y, transpose=transpose)
    norm = da.linalg.norm(output - expected).compute()
    assert norm < tol, '|Ax - Ax_expected| < tol: {:.2e} < {:.2e}'.format(norm, tol)
    return True

def operations_consistent(A, x, y, expect_Ax, expect_ATy):
    symmetric = isinstance(A, linop.DLOSymmetric)
    alpha = np.random.normal(0, 1)
    beta = np.random.normal(0, 1)
    alpha_a = da.random.normal(0, 1, (), chunks=())
    beta_a = da.random.normal(1, 1, (), chunks=())

    expect_gemv = alpha * expect_Ax + beta * y
    expect_gemvT = alpha * expect_ATy + beta * x
    expect_gemvT.persist()
    expect_gemv_array = alpha_a * expect_Ax + beta_a * y
    assert graph_dot_matches_expected(A, x, expect_Ax)
    assert dot_matches_expected(A, x, expect_Ax)
    assert graph_gemv_matches_expected(alpha, A, x, beta, y, expect_gemv)
    assert gemv_matches_expected(alpha, A, x, beta, y, expect_gemv)
    assert gemv_matches_expected(alpha_a, A, x, beta_a, y, expect_gemv_array)

    if not symmetric:
        assert graph_dot_matches_expected(A, y, expect_ATy, transpose=True)
        assert dot_matches_expected(A, y, expect_ATy, transpose=True)
        assert graph_gemv_matches_expected(alpha, A, y, beta, x, expect_gemvT,
                                           transpose=True)
        assert gemv_matches_expected(alpha, A, y, beta, x, expect_gemvT,
                                     transpose=True)

    return True

class AtomsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        m, n, chunks = 100, 50, 10
        self.A = da.random.normal(0, 1 / (m*n)**0.5, (m, n), chunks=chunks)
        self.B = da.random.normal(0, 1 / (m*n)**0.5, (n, m), chunks=chunks)
        self.D = da.random.normal(0, 1 / n**0.5, n, chunks=chunks)
        self.x = x = da.random.normal(0, 1, n, chunks=chunks)
        self.y = y = da.random.normal(0, 1, m, chunks=chunks)
        self.rho = np.random.random()

    def test_array_ops(self):
        # skinny
        Ax = self.A.dot(self.x)
        ATy = self.A.T.dot(self.y)
        assert operations_consistent(self.A, self.x, self.y, Ax, ATy)

        # fat
        By = self.B.dot(self.y)
        BTx = self.B.T.dot(self.x)
        assert operations_consistent(self.B, self.y, self.x, By, BTx)

    def test_DLODense_ops(self):
        A = linop.DLODense(self.A)
        Ax = self.A.dot(self.x)
        ATy = self.A.T.dot(self.y)
        assert operations_consistent(A, self.x, self.y, Ax, ATy)

        # fat
        B = linop.DLODense(self.B)
        By = self.B.dot(self.y)
        BTx = self.B.T.dot(self.x)
        assert operations_consistent(B, self.y, self.x, By, BTx)

    def test_DLODiag_ops(self):
        D = linop.DLODiagonal(self.D)
        Dx = self.D * self.x
        assert operations_consistent(D, self.x, self.x, Dx, Dx)

    def test_DLOGram_ops(self):
        # skinny
        ATA = linop.DLOGram(self.A)
        AAT = linop.DLOGram(self.A, transpose=True)
        ATAx = self.A.T.dot(self.A.dot(self.x))
        AATy = self.A.dot(self.A.T.dot(self.y))
        assert operations_consistent(ATA, self.x, self.x, ATAx, ATAx)
        assert operations_consistent(AAT, self.y, self.y, AATy, AATy)

        # fat
        BBT = linop.DLOGram(self.B)
        BTB = linop.DLOGram(self.B, transpose=False)
        BBTx = self.B.dot(self.B.T.dot(self.x))
        BTBy = self.B.T.dot(self.B.dot(self.y))
        assert operations_consistent(BBT, self.x, self.x, BBTx, BBTx)
        assert operations_consistent(BTB, self.y, self.y, BTBy, BTBy)

    def test_DLORegularizedGram_ops(self):
        rho = self.rho
        ATA = linop.DLORegularizedGram(self.A, regularization=rho)
        AAT = linop.DLORegularizedGram(self.A, regularization=rho, transpose=True)
        rhoATAx = rho * self.x + self.A.T.dot(self.A.dot(self.x))
        rhoAATy = rho * self.y + self.A.dot(self.A.T.dot(self.y))
        assert operations_consistent(ATA, self.x, self.x, rhoATAx, rhoATAx)
        assert operations_consistent(AAT, self.y, self.y, rhoAATy, rhoAATy)

        # fat
        BBT = linop.DLORegularizedGram(self.B, regularization=rho)
        BTB = linop.DLORegularizedGram(self.B, regularization=rho, transpose=False)
        BBTx = rho * self.x + self.B.dot(self.B.T.dot(self.x))
        BTBy = rho * self.y + self.B.T.dot(self.B.dot(self.y))
        assert operations_consistent(BBT, self.x, self.x, BBTx, BBTx)
        assert operations_consistent(BTB, self.y, self.y, BTBy, BTBy)


