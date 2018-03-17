import numpy as np
import dask
import dask.array as da
from . import cones

def dask_matches_numpy(K, x):
    tol_cone = 10**isinstance(K, cones.PositiveSemidefiniteCone)
    xp = cones.project_cone(K, x)
    xd = da.from_array(x, chunks=x.size/2)
    xdp = cones.project_cone(K, xd).compute()
    norm, tol = np.linalg.norm(xp - xdp), 1e-15 * (1 + x.size**0.5) * tol_cone
    assert norm < tol, '|proj_npy - proj_dask| < tol: {} < {}'.format(norm, tol)
    return True

def test_zero_cone():
    m = 10
    K, x = cones.ZeroCone(m), np.random.normal(0, 1, m)
    Kstar = cones.ZeroDualCone(m)
    assert dask_matches_numpy(K, x)
    assert dask_matches_numpy(Kstar, x)
    assert sum(cones.project_cone(K, x)) == 0
    assert sum(cones.project_cone(Kstar, x) - x) == 0

def test_nonnegative_cone():
    m = 10
    K, x = cones.NonnegativeCone(m), np.random.normal(0, 1, m)
    Kstar = cones.NonnegativeDualCone(m)
    assert dask_matches_numpy(K, x)
    assert dask_matches_numpy(Kstar, x)

def test_SOC():
    m = 10
    K, x = cones.SecondOrderCone(m), np.random.normal(0, 1, m)
    Kstar = cones.SecondOrderDualCone(m)
    # cone behavior
    xp = cones.project_cone(K, x)
    xp[0] = 2 * xp[0]
    xpp = cones.project_cone(K, xp)
    assert xpp[0] == xp[0]
    assert sum(xpp[1:] - xp[1:]) == 0
    xp[0] = -1.05 * xp[0]
    xpp = cones.project_cone(K, xp)
    assert sum(xpp) == 0

    # dask matches numpy
    assert dask_matches_numpy(K, x)
    assert dask_matches_numpy(Kstar, x)

def test_PSD_cone():
    m = 10
    mc = 5
    K = cones.PositiveSemidefiniteCone(m)
    Kstar = cones.PositiveSemidefiniteDualCone(m)
    X = np.random.random((m, m))
    X = 1 + X.T.dot(X)
    x = X.reshape(-1)
    projx = cones.project_cone(K, x)
    cones.project_cone(K, da.from_array(x, chunks=mc)).compute() - projx
    assert dask_matches_numpy(K, x)
    assert dask_matches_numpy(Kstar, x)

def test_exp_cone():
    # primal exp cone
    yt = 2.
    xt = 3.2
    zt = 1.05 * yt * np.exp(xt/yt)
    vt = np.array([xt, yt, zt])
    assert cones.v_in_Kexp(vt)
    assert sum(vt - cones.proj_exp_cone(vt)) == 0
    vt = np.array([-2, 0, 2])
    assert cones.v_in_Kexp(vt)
    assert sum(vt - cones.proj_exp_cone(vt)) == 0

    # dual exp cone
    u = -2
    w = 2.5
    v = 1.05 * u * (1 * np.log(-w/u))
    vt = -np.array([u, v, w])
    assert cones.negv_in_Kexp_star(vt)
    assert sum(cones.proj_exp_cone(vt)) == 0
    vt = -np.array([0, 1, 0.5])
    assert cones.negv_in_Kexp_star(vt)
    assert sum(cones.proj_exp_cone(vt)) == 0

    # project to bd(Kexp)
    vt = np.array([0, 0, 0.5])
    assert sum(cones.proj_exp_cone(vt) - vt) == 0
    vt = np.array([-1, -1, 0.5])
    vproj = np.array([-1, 0, 0.5])
    assert sum(cones.proj_exp_cone(vt) - vproj) == 0
    vt = np.array([-1, -1, -0.5])
    vproj = np.array([-1, 0, 0])
    assert sum(cones.proj_exp_cone(vt) - vproj) == 0

    # remaining projections
    yt = 2.
    xt = 3.2
    zt = 1.01 * yt * np.exp(xt/yt)
    vt = np.array([xt, yt, 0.95 * zt])
    vtp = cones.proj_exp_cone(vt)
    assert not cones.v_in_Kexp(vt)
    assert cones.v_in_Kexp(vtp)
    vt = np.array([2 * xt, 2 * yt, zt])
    vtp = cones.proj_exp_cone(vt)
    assert not cones.v_in_Kexp(vt)
    assert cones.v_in_Kexp(vtp)

    m = 10
    mc = 5
    K = cones.ExponentialCone(m)
    Kstar = cones.ExponentialDualCone(m)
    x = np.random.uniform(1, 10, 3 * m)
    projx = cones.project_cone(K, x)
    PX = projx.reshape((-1, 3))
    assert all([cones.v_in_Kexp(PX[i, :]) for i in range(m)])
    assert dask_matches_numpy(K, x)
    assert dask_matches_numpy(Kstar, x)

def test_pow_cone():
    m = 1
    mc = 5
    p = np.random.uniform(-1, 1, m)
    K = cones.PowerCone(p)
    Kstar = cones.PowerDualCone(p)
    x = np.random.uniform(1, 10, 3 * m)
    projx = cones.project_cone(K, x)
    PX = projx.reshape((-1, 3))
    def in_cone(v, p):
        return cones.v_in_Ka(v, p) if p > 0 else cones.negv_in_Ka_star(-v, -p)
    assert all([in_cone(PX[i, :], p[i]) for i in range(m)])
    assert dask_matches_numpy(K, x)
    assert dask_matches_numpy(Kstar, x)