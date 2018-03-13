import numpy as np
import dask
import dask.array as da
import unittest

import scs_dask.linalg.linear_operator as linop
import scs_dask.linalg.atoms as atoms
import scs_dask.linalg.atoms2 as atoms2
import scs_dask.linalg.cg_variants as cg

def cg_reduces_residuals(A):
    options = dict(graph_iters=1, maxiter=1000)
    tol = 1e-5
    b = da.ones(A.shape[0], chunks=A.chunks[0])
    x, _, _ = cg.cg_graph(A, b)
    nrm = da.linalg.norm(b - atoms2.dot(A, x)).compute()
    nrm_tol = tol * (1 + da.linalg.norm(b)).compute()
    return nrm < nrm_tol

def cg_preconditioner_reduces_iterations(A, M):
    options = dict(graph_iters=1, maxiter=1000)
    b = da.ones(A.shape[0], chunks=A.chunks[0])
    _, _, iters = cg.cg_graph(A, b, **options)

    b = da.ones(A.shape[0], chunks=A.chunks[0])
    _, _, iters_pre = cg.cg_graph(A, b, preconditioner=M, **options)
    return iters_pre < iters

def cg_warmstart_reduces_iterations(A):
    options = dict(graph_iters=1, maxiter=1000)
    b = da.ones(A.shape[0], chunks=A.chunks[0])
    x, res, iters = cg.cg_graph(A, b, **options)

    # no iters if solved
    _, _, iters_ws = cg.cg_graph(A, b, x_init=x, **options)
    assert iters_ws == 0

    # iters deterministic
    x_partial, res_partial, iters_partial = cg.cg_graph(A, b, **options)
    _, _, iters_ws = cg.cg_graph(A, b, x_init=x_partial, **options)
    assert iters_partial + iters_ws == iters

    perturb_x = 0.1 * da.mean(x).compute() / (x.size**0.5)
    perturb_b = 0.1 * da.mean(b).compute() / (b.size**0.5)
    xp = x * (1 + da.random.normal(0, perturb_x, x.size, chunks=x.chunks))
    bp = b * (1 + da.random.normal(0, perturb_b, b.size, chunks=b.chunks))

    # nearby b
    _, _, iterp = cg.cg_graph(A, bp, x_init=x, **options)
    assert iters > iterp, '{} > {}'.format(iters, iterp)

    # nearby x0
    _, _, iters_perturb = cg.cg_graph(A, b, x_init=xp, **options)
    assert iters > iterp, '{} > {}'.format(iters, iterp)

    # nearby (b, x0)
    _, _, iters_perturb = cg.cg_graph(A, bp, x_init=xp, **options)
    assert iters > iterp, '{} > {}'.format(iters, iterp)
    return True

class CGTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        m, n, chunks = 1000, 500, 100
        self.rho = rho = np.random.normal(1, 0.5)
        A = np.random.normal(0, 1 / (m*n)**0.5, (m, n))
        B = np.random.normal(0, 1 / (m*n)**0.5, (n, m))
        self.A = da.from_array(A, chunks=chunks).persist()
        self.B = da.from_array(B, chunks=chunks).persist()
        self.dATA = da.from_array(atoms.diag_gram(A), chunks=chunks).persist()
        self.dAAT = da.from_array(atoms.diag_gram(A, transpose=True), chunks=chunks).persist()
        self.dBBT = da.from_array(atoms.diag_gram(B, transpose=True), chunks=chunks).persist()
        self.dBTB = da.from_array(atoms.diag_gram(B), chunks=chunks).persist()

        diagA = 0.5 + np.sqrt(range(1, m + 1))
        Acg = (
            np.eye(m, m, -1)
            + np.eye(m, m, 1)
            + np.eye(m, m, -m/10)
            + np.eye(m, m, m/10)
            + np.diag(diagA))
        Mcg = np.diag(1. / diagA)

        self.Acg = da.from_array(Acg, chunks=chunks).persist()
        self.Mcg = da.from_array(Mcg, chunks=chunks).persist()

    def test_cg_array(self):
        A = self.Acg
        M = self.Mcg
        assert cg_reduces_residuals(A)
        assert cg_preconditioner_reduces_iterations(A, M)
        assert cg_warmstart_reduces_iterations(A)

    def test_cg_DLODense(self):
        A = linop.DLODense(self.Acg)
        M = linop.DLODense(self.Mcg)
        assert cg_reduces_residuals(A)
        assert cg_preconditioner_reduces_iterations(A, M)
        assert cg_warmstart_reduces_iterations(A)

    def test_cg_DLOGram(self):
        ATA = linop.DLOGram(self.A)
        AAT = linop.DLOGram(self.A, transpose=True)
        BBT = linop.DLOGram(self.B)
        BTB = linop.DLOGram(self.B, transpose=False)
        M_ATA = linop.DLODiagonal(self.dATA)
        M_BBT = linop.DLODiagonal(self.dBBT)

        # skinny, full rank Gram
        assert cg_reduces_residuals(ATA)
        assert cg_warmstart_reduces_iterations(ATA)

        # skinny, rank-deficient Gram
        assert not cg_reduces_residuals(AAT)

        # fat, full rank Gram
        assert cg_reduces_residuals(BBT)
        assert cg_warmstart_reduces_iterations(BBT)

        # fat, rank-deficient Gram
        assert not cg_reduces_residuals(BTB)

    def test_cg_DLORegularizedGram(self):
        ATA = linop.DLORegularizedGram(self.A)
        AAT = linop.DLORegularizedGram(self.A, transpose=True)
        BBT = linop.DLORegularizedGram(self.B)
        BTB = linop.DLORegularizedGram(self.B, transpose=False)

        M_ATA = linop.DLODiagonal((1. / (self.rho + self.dATA)).persist())
        M_AAT = linop.DLODiagonal((1. / (self.rho + self.dAAT)).persist())
        M_BBT = linop.DLODiagonal((1. / (self.rho + self.dBBT)).persist())
        M_BTB = linop.DLODiagonal((1. / (self.rho + self.dBTB)).persist())

        for A, M in ((ATA, M_ATA), (AAT, M_AAT), (BBT, M_BBT), (BTB, M_BTB)):
            assert cg_reduces_residuals(A)
            assert cg_warmstart_reduces_iterations(A)

class CGLSTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        m, n, chunks = 1000, 500, 100
        self.A = A = da.random.normal(0, 1 / (m*n)**0.5, (m, n), chunks=chunks)
        self.B = B = da.random.normal(0, 1 / (m*n)**0.5, (n, m), chunks=chunks)

    def test_cgls_skinny(self):
        tol = 1e5
        A = self.A
        x0 = da.ones(A.shape[1], chunks=(A.chunks[1],))
        b = atoms2.dot(A, x0)
        tol_cgls = tol * da.linalg.norm(b) / da.linalg.norm(x0)
        x, res, iters = cg.cgls(A, b, 0., tol=tol_cgls)
        r = atoms2.dot(A, x) - b
        assert da.linalg.norm(r).compute() < tol * (1 + x.size**0.5)

    def test_cgls_fat(self):
        tol = 1e5
        A = self.B
        x0 = da.ones(A.shape[1], chunks=(A.chunks[1],))
        b = atoms2.dot(A, x0)
        tol_cgls = tol * da.linalg.norm(b) / da.linalg.norm(x0)
        x_ln, res, iters = cg.cgls(A, b, 0., tol=tol_cgls)
        r = atoms2.dot(A, x_ln) - b
        assert da.linalg.norm(r).compute() < tol * (1 + x0.size**0.5)
        assert (x0 - x_ln).dot(x_ln).compute() < tol * (1 + x0.size**0.5)
