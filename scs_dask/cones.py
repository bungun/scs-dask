import operator
import functools
import numpy as np
import dask
import dask.array as da
import multipledispatch

namespace_cones = dict()
dispatch = functools.partial(multipledispatch.dispatch, namespace=namespace_cones)

CONE_TOL = 1e-8
CONE_THRESH = 1e-6
EXP_CONE_MAX_ITERS = 100
POW_CONE_MAX_ITERS = 20

class ConvexCone(object):
    def __init__(self, dimension):
        self.dim = int(dimension)

    def __dask_tokenize__(self):
        return (type(self), self.dim)

class ZeroCone(ConvexCone):
    pass

class ZeroDualCone(ConvexCone):
    pass

class NonnegativeCone(ConvexCone):
    def __init__(self, dimension):
        ConvexCone.__init__(self, dimension)

NonnegativeDualCone = NonnegativeCone

class SecondOrderCone(ConvexCone):
    pass

SecondOrderDualCone = SecondOrderCone

class PositiveSemidefiniteCone(ConvexCone):
    pass

PositiveSemidefiniteDualCone = PositiveSemidefiniteCone
PSDCone = PositiveSemidefiniteCone

# exp(x)
# perspective: y*exp(x/y), y > 0
# K = epi[(y*exp(x/y)), y > 0] = {(x, y, z) | y*exp(x/y) >= z, y > 0}
# K_exp = cl(K) = K \cup {(x, 0, z) \in R^3| x =< 0, z >= 0}
class ExponentialCone(ConvexCone):
    pass

# K_exp^* = {(u, v, w) \in R_- \times R \times R_+ | -ulog(-u/w) + u - v <= 0}
#           \cup {(0, v, w) | v >= 0, w >= 0}
class ExponentialDualCone(ConvexCone):
    pass

class PowerCone(ConvexCone):
    def __init__(self, powers):
        powers = np.array(powers)
        assert all(abs(powers) <= 1)
        ConvexCone.__init__(self, len(powers))
        self.powers = powers

class PowerDualCone(PowerCone):
    def __init__(self, powers):
        PowerCone.__init__(self, -1 * np.array(powers))

CONES = (
    'Zero',
    'Nonnegative',
    'SecondOrder',
    'PositiveSemidefinite',
    'Exponential',
    'Power')
DUAL_CONES = {eval(nm + 'Cone'): eval(nm + 'DualCone') for nm in CONES}
DUAL_CONES.update({dc: pc for (pc, dc) in DUAL_CONES.items()})

def K_to_Kstar(K):
    return DUAL_CONES[type(K)](K.dim)

@dispatch(ZeroCone, (np.ndarray, da.Array))
def project_cone(K, x):
    return 0 * x

@dispatch(ZeroDualCone, (np.ndarray, da.Array))
def project_cone(K, x):
    return 1 * x

@dispatch(NonnegativeCone, np.ndarray)
def project_cone(K, x):
    return np.maximum(x, 0)

@dispatch(NonnegativeCone, da.Array)
def project_cone(K, x):
    return da.maximum(x, 0)

@dispatch(SecondOrderCone, np.ndarray)
def project_cone(K, x):
    s, v = x[0, ...], x[1:, ...]
    norm_v = np.linalg.norm(v)
    if norm_v <= -s:
        return 0 * x
    elif norm_v <= s:
        return 1 * x
    else:
        return 0.5*(1 + s/norm_v) * np.hstack(([norm_v], v))

@dispatch(SecondOrderCone, da.Array)
def project_cone(K, x):
    s = x[0].compute()
    v = x[1:]
    norm_v = da.linalg.norm(v).compute()

    if norm_v <= -s:
        projx = 0 * x
    elif norm_v <= s:
        projx = 1 * x
    else:
        scal = 0.5 * (1 + s/norm_v)
        s = da.from_array(np.array([norm_v]), chunks=(1,))
        projx = scal * da.hstack((s, v))
    return projx

@dispatch(PositiveSemidefiniteCone, np.ndarray)
def project_cone(K, x):
    L, Q = np.linalg.eigh(np.reshape(x, (K.dim, K.dim)))
    return (Q.dot(np.maximum(0, L).reshape(-1, 1) * Q.T)).reshape(-1)

@dispatch(PositiveSemidefiniteCone, da.Array)
def project_cone(K, x):
    assert x.size == K.dim**2, 'input dimension compatible'
    chunks = x[:K.dim].chunks[0]
    X = da.reshape(x, (K.dim, K.dim)).rechunk((chunks, (K.dim,)))
    U, S, V = da.linalg.svd(da.reshape(x, (K.dim, K.dim)))
    return U.dot(da.maximum(0, S).reshape(-1, 1) * V).reshape(-1)

def exp_solve_fixed_rho(v0, v, rho):
    """ Solve scalarized cone projection problem for fixed Lagrange multiplier rho > 0.

        We can solve

            argmin g(x, y, z) = 0.5(x-x0)^2 + 0.5 (y-y0)^2 + 0.5(z-z0)^2 + x0 - rho * ylog(z/y)

        by using the optimality conditions:

            g_x = 0 = x - x0 + rho
            g_y = 0 = y - y0 - rho log(z/y) + rho
            g_z = 0 = z - z0 - rho y/z.

        Substitute rho y = z(z-z0) in [2]:

            g_y = 0 = (1/rho)z(z-z0) -y0 + rho log((z-z0)/rho) + rho,
                    = (1/rho^2)z(z-z0) -y0/rho + log((z-z0)/rho) + 1,

        Let t = z-z0 and define f: R->R as

            f(t) = t(t+z0)/rho^2 - y0/rho + log(t/rho) + 1.

        Find t such that f(t) = 0 by Newton Raphson.

        Take z_init = 0 -> t_init = 0 - z0. Take t_init = max(epsilon, -z0) for domain of log().

        Solution v^star = (x^star, y^star, z^star) is then:

            z^star = t^star + z0
            y^star = (1/rho) * z^star(z^star - z0) = (1/rho) * t^star * z^star
            x^star = x0 - rho
    """
    x0, y0, z0 = v0
    ti = max(-z0, 1e-6)
    f = lambda t: (1/rho**2)*t*(t+z0) - y0/rho + np.log(t/rho) + 1
    df = lambda t: (2*t + z0)/rho**2 + 1/t
    for i in range(EXP_CONE_MAX_ITERS):
        fi = f(ti)
        ti -= fi / df(ti)
        if (ti <= -z0):
            return 0 * z0
        elif (ti <= 0):
            return z0
        elif abs(fi) < CONE_TOL:
            break
    t_star = ti
    z_star = t_star + z0

    v[2] = t_star
    v[1] = t_star * z_star / rho
    v[0] = x0 - rho
    return v

def exp_grad_rho(v0, v, rho):
    """
    Evaluate gradient of L(v, rho) w.r.t to rho.

    Set v = argmin L(v) = |v-v0|_2 + rho g(v) for fixed rho > 0.
    Let (r, s, t) = v. Then

        L_rho = g(v) = r - s log(t/s), s > epsilon
                     = 0             , otherwise
    """
    v = exp_solve_fixed_rho(v0, v, rho)
    r, s, t = v
    if (s <= 1e-12):
        g = r
    else:
        g = r + s * np.log(s/t)
    return g, v

def exp_rho_bounds(v0, v):
    """
    Initialize bounds for bisection search for optimal rho.

    Set lower bound = 0
    Find upper bound by doubling rho until minimizing

        |v - v0|_2^2 + rho g(v)

    yields a solution v such that g(v) < 0.
    """
    lb_rho = 0.
    ub_rho = 0.125
    g, v = exp_grad_rho(v0, v, ub_rho)
    while g > 0:
        lb_rho = ub_rho
        ub_rho *= 2
        g, v = exp_grad_rho(v0, v, ub_rho)
    return lb_rho, ub_rho, v

# r * exp(s/r) <= t
def v_in_Kexp(v):
    """
    Test whether v lies in exponential primal cone
    """
    r, s, t = v
    return ((r <= 0 and s == 0 and t >= 0)
             or (s > 0 and s*np.exp(r/s) - t <= CONE_THRESH))

# -r * exp(s/r) <= t*exp(1)
def negv_in_Kexp_star(v):
    """
    Test whether v lies in exponential dual cone
    """
    r, s, t = -v
    return ((r == 0 and s >= 0 and t >= 0)
            or (r < 0 and t >=0 and -r * np.exp(s/r) - np.exp(1) * t <= CONE_THRESH))

def v_projects_to_bd_Kexp(v):
    """
    Test whether projection of v onto exponential primal cone lies in {(x, 0, z) | x <= 0, z >= 0}.
    """
    return v[0] < 0 and v[1] < 0

def proj_exp_cone(v):
    """
    Project v = (r0, s0, t0) onto closure of set {(r, s, t) | s*exp(r/s) - t <= 0}.

    If v in primal cone, return v.
    If v in polar cone (-v in dual cone), return 0
    If r0, s0 < 0, apply analytic projection rule.
    Otherwise, project by solving the problem

        argmin      |(x, y, z) - (x0, y0, z0)|_2^2
        subject to  y exp(x0/y) - z <= 0.

    The exponential cone constraint can also be expressed as

        x - ylog(z/y) <= 0.

    We can minimize the equivalent unconstrained problem

        min_{v, rho} L(v, rho) = min_{v, rho} |v - v0| + rho g(v),

    where g(v) <= 0 encodes the exponential cone.

    In particular, we search for an optimal rho by bisection;
    this generates a series of subproblems wherein we solve
    an unconstrained minimization over (r, s, t) for fixed rho.

    The bisection search policy is as follows:

        if grad_rho(L) = g(v) < 0, v is feasible, try smaller rho
        if grad_rho(L) = g(v) > 0, v infeasible, try larger rho.

    Initialize with interval [rho_lower, rho_upper]. We can
    take rho_lower = 0, and initialize rho_upper to be any feasible
    value of v.
    """
    if v_in_Kexp(v):
        projv = 1 * v
    elif negv_in_Kexp_star(v):
        projv = 0 * v
    elif v_projects_to_bd_Kexp(v):
        projv = np.array([v[0], 0, max(v[2], 0)])
    else:
        tol = CONE_TOL
        projv = 0 * v
        lb, ub, projv = exp_rho_bounds(v, projv)
        for i in range(EXP_CONE_MAX_ITERS):
            rho = 0.5 * (lb + ub)
            g, projv = exp_grad_rho(v, projv, rho)
            if g > 0:
                lb = rho
            else:
                ub = rho
            if ub - lb < tol:
                break
    return projv

def proj_exp_dual_cone(v):
    """
    Project v onto exponential dual cone, via Moreau decomposition
    """
    return v + proj_exp_cone(-v)

@dispatch(ExponentialCone, np.ndarray)
def project_cone(K, x):
    assert x.size == 3 * K.dim, 'input dimension compatible'
    return np.hstack([proj_exp_cone(x[3*i:3*(i+1)]) for i in range(K.dim)])

@dispatch(ExponentialCone, da.Array)
def project_cone(K, x):
    assert x.size == 3 * K.dim, 'input dimension compatible'
    return da.map_blocks(proj_exp_dual_cone, x.rechunk(3)).rechunk(chunks=x.chunks)

@dispatch(ExponentialDualCone, np.ndarray)
def project_cone(K, x):
    assert x.size == 3 * K.dim, 'input dimension compatible'
    return np.hstack([proj_exp_dual_cone(x[3*i:3*(i+1)]) for i in range(K.dim)])

@dispatch(ExponentialDualCone, da.Array)
def project_cone(K, x):
    assert x.size == 3 * K.dim, 'input dimension compatible'
    return da.map_blocks(proj_exp_dual_cone, x.rechunk(3)).rechunk(chunks=x.chunks)

def v_in_Ka(v, a):
    return (v[0] >= 0
            and v[1] >= 0
            and CONE_THRESH + v[0]**a * v[1]**(1-a) >= abs(v[2]))
def negv_in_Ka_star(v, a):
    return (v[0] <= 0
            and v[1] <= 0
            and CONE_THRESH + (-v[0])**a * (-v[1])**(1-a) >= abs(v[2]) * a**a * (1-a)**(1-a))

def pow_calc_x(r, xh, rh, a):
    return max(1e-12, 0.5 * (xh + np.sqrt(xh**2 + 4*a*(rh-r)*r)))
def pow_calc_dxdr(x, xh, rh, r, a):
    return a * (rh - 2*r) / (2*x - xh)
def pow_calc_f(x, y, r, a):
    return pow(x, a) * pow(y, (1-a)) - r
def pow_calc_fp(x, y, dxdr, dydr, a):
    return pow(x, a)*pow(y, (1-a))*(a*dxdr/x + (1-a)*dydr/y) - 1

def proj_power(v, a):
    assert len(v) == 3
    v = np.array(v)
    if v_in_Ka(v, a):
        projv = v
    elif negv_in_Ka_star(v, a):
        projv = 0 * v
    else:
        xx, yy, rr = v[0], v[1], abs(v[2])
        xi, yi, ri = 0., 0., rr/2
        for i in range(POW_CONE_MAX_ITERS):
            xi = pow_calc_x(ri, xx, rr, a)
            yi = pow_calc_x(ri, yy, rr, 1-a)
            fi = pow_calc_f(xi, yi, ri, a)
            if abs(fi) < CONE_TOL:
                break
            dxdr = pow_calc_dxdr(xi, xx, rr, ri, a)
            dydr = pow_calc_dxdr(yi, yy, rr, ri, 1-a)
            fp = pow_calc_fp(xi, yi, dxdr, dydr, a)
            ri = min(rr, max(ri - fi/fp, 0))
        projv = np.array([xi, yi, -ri if v[2] < 0 else ri])
    return projv

def proj_power_dual(v, a):
    v = np.array(v)
    return v + proj_power(-v, a)

def proj_pow_cone(v, a):
    # a > 0: primal cone
    # a <= 0: dual cone via Moreau decomposition x = prox_K(x) + prox_Kstar(-x)
    return proj_power(v, a) if a > 0 else proj_power_dual(v, -a)

# {(x,y,z) | x^a * y^(1-a) >= |z|, x>=0, y>=0}
@dispatch(PowerCone, np.ndarray)
def project_cone(K, x):
    assert x.size == 3 * K.dim, 'input dimension compatible'
    return np.hstack([proj_pow_cone(x[3*i:3*(i+1)], K.powers[i]) for i in range(K.dim)])

@dispatch(PowerCone, da.Array)
def project_cone(K, x):
    assert x.shape[0] == x.size and x.size == 3 * K.dim, 'input dimension compatible'
    x3 = x.rechunk(chunks=3)
    dsk = dict()
    token = 'project-power-cone-' + dask.base.tokenize(K, x)
    for i in range(x3.numblocks[0]):
        dsk[(token, i)] = (proj_pow_cone, (x3.name, i), K.powers[i])
    projx = da.Array(
            dask.sharedict.merge(dsk, x3.dask), token, shape=x.shape,
            chunks=x3.chunks, dtype=x.dtype)
    return projx.rechunk(chunks=x.chunks)