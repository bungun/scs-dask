import time
import operator
import functools
import dask
import dask.array as da

import scs_dask.linalg.linear_operator as linop
import scs_dask.linalg.atoms2 as atoms2


def iter_options(graph_iters=1, verbose=0, print_iters=0, time_iters=0):
    graph_iters = max(1, int(graph_iters))
    time_iters = max(0, int(time_iters))
    if int(print_iters) < 1 and verbose > 0:
        print_iters = max(0, max(int(print_iters), int(10**(3 - verbose))))
    if print_iters > 0:
        print_iters = max(print_iters, graph_iters)
    return graph_iters, print_iters, time_iters

def cg_initialize(A, b, x_init=None):
    if x_init is None:
        x = 0 * b
    else:
        x = 1 * x_init
    r = A.dot(x) - b
    p = 1 * r
    x, r, p = dask.persist(x, r, p)
    return x, r, p

def cg_iterate(A, state, persist=True):
    ox, or_, op = state
    Ap = A.dot(op)
    alpha = or_.dot(or_) / op.dot(Ap)
    x = ox + alpha * op
    r = or_ - alpha * Ap
    p = r + op * r.dot(r) / or_.dot(or_)
    x, r, p = dask.optimize(x, r, p)
    if persist:
        x, r, p = dask.persist(x, r, p, optimize_graph=False)
    return x, r, p

def cg_residual(state, compute=True):
    _, r, _ = state
    res = da.linalg.norm(r)
    if compute:
        (res,) = dask.compute(res)
    return res

def cg(A, b, tol=1e-5, maxiter=500, **options):
    graph_iters, print_iters, time_iters = iter_options(**options)
    graph_iters = max(1, int(graph_iters))
    state = cg_initialize(A, b)
    start = time.time()
    for i in range(1, maxiter + 1):
        calculate = bool(i % graph_iters == 0)
        state = cg_iterate(A, state, persist=calculate)
        res = cg_residual(state, compute=calculate)
        if i % 10 == 0:
            print i, time.time() - start
            start = time.time()
        if calculate:
            if i % 10 == 0:
                print '\t', i, res
            if res < tol:
                break
    x, _, _ = state
    res = cg_residual(state, compute=True)
    (x,) = dask.persist(x)
    return x, res, i

def cg_init_dsk(A, b, state0, x_init=None):
    x0, r0, p0 = map(lambda nm: nm + '-' + state0, ('x', 'r', 'p'))
    x0_vec = b if x_init is None else x_init
    x0_scal = 0 if x_init is None else 1
    def init_x(veci): return x0_scal * veci
    def init_p(ri): return 1 * ri
    dsk = dict()
    vblocks, hblocks = A.numblocks
    for i in range(vblocks):
        dsk[(x0, i)] = (init_x, (x0_vec.name, i))
        dsk[(r0, i)] = (operator.sub,
                (
                        da.core.dotmany,
                        [(A.name, i, j) for j in range(hblocks)],
                        [(x0, j) for j in range(hblocks)]),
                (b.name, i))
        dsk[(p0, i)] = (init_p, (r0, i))
    return dsk

def cg_iterate_dsk(A, state0, state1):
    Ap, pAp = 'Ap-' + state0, 'pAp-' + state0
    x0, r0, p0, gamma0 = map(lambda nm: nm + '-' + state0, ('x', 'r', 'p', 'gamma'))
    x1, r1, p1, gamma1 = map(lambda nm: nm + '-' + state1, ('x', 'r', 'p', 'gamma'))
    def update_x(x, gamma, pAp, p): return x + (gamma / pAp) * p
    def update_r(r, gamma, pAp, Ap): return r - (gamma / pAp) * Ap
    def update_p(p, gamma, gamma_next, r): return r + (gamma_next / gamma) * p
    dsk = dict()
    vblocks, hblocks = A.numblocks
    for i in range(vblocks):
        dsk[(Ap, i)] = (da.core.dotmany,
                        [(A.name, i, j) for j in range(hblocks)],
                        [(p0, j) for j in range(hblocks)])
    dsk[gamma0] = (da.core.dotmany,
                   [(r0, i) for i in range(vblocks)],
                   [(r0, i) for i in range(vblocks)])
    dsk[pAp] = (da.core.dotmany,
                [(p0, i) for i in range(vblocks)],
                [(Ap, i) for i in range(vblocks)])
    for i in range(vblocks):
        dsk[(x1, i)] = (update_x, (x0, i), gamma0, pAp, (p0, i))
        dsk[(r1, i)] = (update_r, (r0, i), gamma0, pAp, (Ap, i))
        dsk[(p1, i)] = (update_p, (p0, i), gamma0, gamma1, (r1, i))
    dsk[gamma1] = (da.core.dotmany,
                   [(r1, i) for i in range(vblocks)],
                   [(r1, i) for i in range(vblocks)])
    return dsk

# def cg_calcs_proto(shape, chunks, dtype, dsk, key, optimize=False):
#     x = da.Array(dsk, 'x-' + key, shape=shape, chunks=chunks, dtype=dtype)
#     r = da.Array(dsk, 'r-' + key, shape=shape, chunks=chunks, dtype=dtype)
#     p = da.Array(dsk, 'p-' + key, shape=shape, chunks=chunks, dtype=dtype)
#     if optimize:
#         (x, r, p) = dask.optimize(x, r, p)
#     (x, r, p) = dask.persist(x, r, p, optimize_graph=False, traverse=False)
#     (res,) = dask.compute(da.linalg.norm(r))
#     return x, r, p, res

def cg_dsk(A, b, tol=1e-5, maxiter=500, **options):
    cg_calcs = functools.partial(cg_calcs_proto, b.shape, b.chunks, b.dtype)
    graph_iters, print_iters, time_iters = iter_options(**options)
    key_init = 'cg-iter0'
    dsk = dask.sharedict.merge(A.dask, b.dask, cg_init_dsk(A, b, key_init))
    x, r, p, res = cg_calcs(dsk, key_init)
    if time_iters > 0:
        start = time.time()
    dsk = dict()
    for i in range(1, maxiter + 1):
        key0 = 'cg-iter{}'.format(i - 1)
        key1 = 'cg-iter{}'.format(i)
        calculate = bool(i % graph_iters == 0)
        dsk.update(cg_iterate_dsk(A, key0, key1))
        if calculate:
            dsk = dask.sharedict.merge(A.dask, x.dask, r.dask, p.dask, dsk)
            x, r, p, res = cg_calcs(dsk, key1)
            dsk = dict()
            if print_iters > 0 and i % print_iters == 0:
                print '\t\t\t{}: residual = {:.1e}'.format(i, res)
            if res < tol:
                break
        if time_iters > 0 and i % time_iters == 0:
            print '{}: {:.1e} seconds'.format(i, time.time() - start)
            start = time.time()
    if i == maxiter:
        dsk = dask.sharedict.merge(A.dask, x.dask, r.dask, p.dask, dsk)
        x, _, _, res = cg_calcs(dsk, key1)
    return x, res, i

def list_blocks(nm, blocks=0): return [(nm, i) for i in range(blocks)]
def dict_blocks(dictionary, nm, blocks=0): return [dictionary[(nm, i)] for i in range(blocks)]

def cg_init_graph(A, b, state0, x_init=None, M=None, M12=None):
    if x_init is not None and M is not None and M12 is None:
        raise ValueError('warm start (x0) and preconditioner (M) given, M^{1/2} required')
    x0, r0, p0 = map(lambda nm: nm + '-' + state0, ('x', 'r', 'p'))
    scal_x = float(x_init is not None)
    vec_x = b if x_init is None else x_init
    def init_x(x_or_b): return scal_x * x_or_b
    def init_p(ri): return 1 * ri
    dsk = dict()
    vblocks, hblocks = A.numblocks
    for i in range(vblocks):
        dsk[(x0, i)] = (init_x, (vec_x.name, i))
    dsk.update(atoms2.graph_gemv(-1, A, x0, 1, b.name, r0))
    if M is None:
        for i in range(vblocks):
            dsk[(p0, i)] = (init_p, (r0, i))
    else:
        dsk.update(atoms2.graph_dot(M, r0, p0))
    return dsk

def cg_iterate_graph(A, state0, state1, M=None):
    Ap, pAp = 'Ap-' + state0, 'pAp-' + state0
    x0, r0, p0, rMr0 = map(lambda nm: nm + '-' + state0, ('x', 'r', 'p', 'rMr'))
    x1, r1, p1, rMr1 = map(lambda nm: nm + '-' + state1, ('x', 'r', 'p', 'rMr'))
    def update_x(x, rMr, pAp, p): return x + (rMr / pAp) * p
    def update_r(r, rMr, pAp, Ap): return r - (rMr / pAp) * Ap
    def update_p(p, rMr, rMr_next, Mr): return Mr + (rMr_next / rMr) * p
    dsk = dict()
    vblocks, hblocks = A.numblocks
    get_blocks = functools.partial(list_blocks, blocks=vblocks)
    get_blocks_d = functools.partial(dict_blocks, blocks=vblocks)
    dsk.update(atoms2.graph_dot(A, p0, Ap))
    if M is None:
        dsk_Mr0 = {('Mr0', i): (r0, i) for i in range(vblocks)}
        dsk_Mr1 = {('Mr1', i): (r1, i) for i in range(vblocks)}
    else:
        dsk_Mr0 = atoms2.graph_dot(M, r0, 'Mr0')
        dsk_Mr1 = atoms2.graph_dot(M, r1, 'Mr1')
    dsk[rMr0] = (da.core.dotmany, get_blocks(r0), get_blocks_d(dsk_Mr0, 'Mr0'))
    dsk[rMr1] = (da.core.dotmany, get_blocks(r1), get_blocks_d(dsk_Mr1, 'Mr1'))

    dsk[pAp] = (da.core.dotmany, get_blocks(p0), get_blocks(Ap))
    for i in range(vblocks):
        dsk[(x1, i)] = (update_x, (x0, i), rMr0, pAp, (p0, i))
        dsk[(r1, i)] = (update_r, (r0, i), rMr0, pAp, (Ap, i))
        dsk[(p1, i)] = (update_p, (p0, i), rMr0, rMr1, dsk_Mr1[('Mr1', i)])
    return dsk

def cg_calcs_proto(shape, chunks, dtype, dsk, key, optimize=False, **options):
    if options.pop('finish', False):
        dsk_final = dict()
        key_final = options.pop('name', 'cg-output')
        for i in range(len(chunks[0])):
            dsk_final[('x-' + key_final, i)] = ('x-' + key, i)
            dsk_final[('r-' + key_final, i)] = ('r-' + key, i)
            dsk_final[('p-' + key_final, i)] = ('p-' + key, i)
            dsk = dask.sharedict.merge(dsk, dsk_final)
        key = key_final
    x = da.Array(dsk, 'x-' + key, shape=shape, chunks=chunks, dtype=dtype)
    r = da.Array(dsk, 'r-' + key, shape=shape, chunks=chunks, dtype=dtype)
    p = da.Array(dsk, 'p-' + key, shape=shape, chunks=chunks, dtype=dtype)
    if optimize:
        (x, r, p) = dask.optimize(x, r, p)
    (x, r, p) = dask.persist(x, r, p, optimize_graph=False, traverse=False)
    (res,) = dask.compute(da.linalg.norm(r))
    return x, r, p, res

def cg_graph(A, b, preconditioner=None, x_init=None, tol=1e-5, maxiter=500, **options):
    token = dask.base.tokenize(A, b, preconditioner, x_init, tol, maxiter, **options)
    M = preconditioner
    M12 = options.pop('preconditioner12', None)
    optimize = options.pop('optimize', False)
    cg_calcs = functools.partial(cg_calcs_proto, b.shape, b.chunks, b.dtype, optimize=optimize)
    graph_iters, print_iters, time_iters = iter_options(**options)
    key_init = 'cg-iter0-{}'.format(token)
    dsk = cg_init_graph(A, b, key_init, x_init=x_init, M=M, M12=M12)
    dsks = [A.dask, b.dask, dsk]
    dsks += [x_init.dask] if x_init is not None else []
    dsks += [M.dask] if M is not None else []
    x, r, p, res0 = cg_calcs(dask.sharedict.merge(*dsks), key_init, **options)
    if res0 < tol or maxiter == 0:
        return x, res0, 0

    if time_iters > 0:
        start = time.time()
    dsk = dict()
    for i in range(1, maxiter + 1):
        key0 = 'cg-iter{}-{}'.format(i - 1, token)
        key1 = 'cg-iter{}-{}'.format(i, token)
        calculate = bool(i % graph_iters == 0)
        dsk.update(cg_iterate_graph(A, key0, key1, M=M))
        if calculate:
            dsks = [A.dask, x.dask, r.dask, p.dask, dsk]
            dsks += [M.dask] if M is not None else []
            x, r, p, res = cg_calcs(dask.sharedict.merge(*dsks), key1)
            dsk = dict()
            if print_iters > 0 and i % print_iters == 0:
                print '\t\t\t{}: residual = {:.1e}'.format(i, res)
            if res < tol:
                break
        if time_iters > 0 and i % time_iters == 0:
            print '{}: {:.1e} seconds'.format(i, time.time() - start)
            start = time.time()
    dsks = [x.dask, r.dask, p.dask]
    dsks += [A.dask, dsk] if i == maxiter else []
    dsks += [M.dask] if (i == maxiter and M is not None) else []
    x, _, _, res = cg_calcs(dask.sharedict.merge(*dsks), key1, finish=True)
    return x, res, i

def cgls(A, b, rho, **options):
    b_hat = atoms2.dot(A, b, transpose=True)
    A_hat = linop.DLORegularizedGram(A, regularization=rho, transpose=False)
    x, _, iters = cg_graph(A_hat, b_hat, **options)
    res = da.linalg.norm(atoms2.dot(A, x) - b).compute()
    return x, res, iters

