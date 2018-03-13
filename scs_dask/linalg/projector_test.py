import dask.array as da
from scs_dask.linalg import projector

def test_cg_projector_skinny():
    mc = 1000
    m, n = 11 * mc, 10 * mc
    A_skinny = da.random.normal(0, 1 / (m * n)**0.5, (m , n), chunks=mc)
    xrand = da.random.random(A_skinny.shape[1], chunks=mc)
    yrand = da.random.random(A_skinny.shape[0], chunks=mc)
    xo, yo, res, iters = projector.cg_project(A_skinny, xrand, yrand)
    xoo, yoo, res_, iters_ = projector.cg_project(A_skinny, xo, yo, x_init=xo)
    print iters
    assert iters_ == 0
    assert da.linalg.norm(xo - xoo) < 1e-15 * (1 + xo.size**0.5)

def test_cg_projector_fat():
    mc = 1000
    m, n = 10 * mc, 11 * mc
    A_fat = da.random.normal(0, 1 / (m * n)**0.5, (m , n), chunks=mc)
    xrand = da.random.random(A_fat.shape[1], chunks=mc)
    yrand = da.random.random(A_fat.shape[0], chunks=mc)
    xo, yo, res, iters = projector.cg_project(A_fat, xrand, yrand)
    xoo, yoo, res_, iters_ = projector.cg_project(A_fat, xo, yo, x_init=xo)
    print iters
    assert iters_ == 0
    assert da.linalg.norm(xo - xoo) < 1e-15 * (1 + xo.size**0.5)

def test_cgls_projector_skinny():
    mc = 1000
    m, n = 11 * mc, 10 * mc
    A_skinny = da.random.normal(0, 1 / (m * n)**0.5, (m , n), chunks=mc)
    xrand = da.random.random(A_skinny.shape[1], chunks=mc)
    yrand = da.random.random(A_skinny.shape[0], chunks=mc)
    xo, yo, res, iters = projector.cgls_project(A_skinny, xrand, yrand)
    xoo, yoo, res_, iters_ = projector.cgls_project(A_skinny, xo, yo, x_init=xo)
    print iters
    assert iters_ == 0
    assert da.linalg.norm(xo - xoo) < 1e-15 * (1 + xo.size**0.5)

def test_cgls_projector_fat():
    mc = 1000
    m, n = 10 * mc, 11 * mc
    A_fat = da.random.normal(0, 1 / (m * n)**0.5, (m , n), chunks=mc)
    xrand = da.random.random(A_fat.shape[1], chunks=mc)
    yrand = da.random.random(A_fat.shape[0], chunks=mc)
    xo, yo, res, iters = projector.cgls_project(A_fat, xrand, yrand)
    xoo, yoo, res_, iters_ = projector.cgls_project(A_fat, xo, yo, x_init=xo)
    print iters
    assert iters_ == 0
    assert da.linalg.norm(xo - xoo) < 1e-15 * (1 + xo.size**0.5)
