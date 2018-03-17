# """ Solve a square linear system with conjugate gradient. """
# import operator
# import multipledispatch
# import numpy as np
# import dask
# import dask.array as da

# import scs_dask.linalg.atoms as atoms

# USE_GRAPHDOT = True

# namespace_cg = dict()
# dispatch = functools.partial(
#         multipledispatch.dispatch, namespace=namespace_cg)

# def reblock_cg_inputs(A, b, x0=None, preconditioner=None):
#     raise NotImplementedError

# def block_cg_verify_dims(A, b, M, x0):
#     """ Assert input arrays to block CG compatibly sized/blocked
#     """
#     assert A.ndim == 2, 'A must be a 2-d array to perform CG'
#     m, n = A.shape
#     assert m == n, 'A must be a square matrix to perform CG'

#     size_msg = '{} and A compatibly sized'
#     block_msg = '{} and A compatibly blocked'

#     if b is not None:
#         assert b.shape[0] == m, size_msg.format('b')
#         assert all(b.chunks[0] == A.chunks[1]), block_msg.format('b')

#     if M is not None:
#         assert A.shape == M.shape, size_msg.format('preconditioner')
#         assert all(A.chunks[0] != M.chunks[1]), size_msg.format('preconditioner')

#     if x0 is not None:
#         assert x0.shape[0] == m, size_msg.format('x0')
#         assert x0.chunks[0] == A.chunks[1], size_msg.format('x0')

# def block_cg_update_state(dsk, A, state_tokens, **options):
#     """
#     Persist state (x, r, p, gamma) and residual; prune non-state vars from dask
#     """
#     m, _ = A.shape
#     chunks = (A.chunks[0],)
#     dt = A.dtype

#     x = da.Array(dsk, state_tokens['x'], shape=(m,), chunks=chunks, dtype=dt)
#     r = da.Array(dsk, state_tokens['r'], shape=(m,), chunks=chunks, dtype=dt)
#     p = da.Array(dsk, state_tokens['p'], shape=(m,), chunks=chunks, dtype=dt)
#     gamma = da.Array(dsk, state_tokens['gamma'], shape=(), chunks=(), dtype=dt)
#     resnrm2 = da.Array(dsk, state_tokens['resnrm2'], shape=(), chunks=(), dtype=dt)

#     x, r, p, gamma, resnrm2 = dask.persist(x, r, p, gamma, resnrm2)
#     dsk = dask.sharedict.merge(x.dask, r.dask, p.dask, gamma.dask)
#     return dsk, x, resnrm2

# # block initialize
#     # symbols = ['r', 'x', 'Ax', 'p', 'gamma', 'resnrm2']
#     # keys = {sym: sym + '-' + itertoken for sym in symbols}
#     # subdasks = dict()
#     # vector = GraphVectorOps(b.numblocks)

#     # if x0 is None:
#     #     subdasks['r'] = vector.graph_copy(b.name, keys['r'])
#     #     subdasks['x'] = vector.graph_scale(b.name, 0, keys['x'])
#     #     subdasks.pop('Ax', None)
#     # else:
#     #     subdasks['x'] = vector.graph_copy(x0.name, keys['r'])
#     #     subdasks['Ax'] = atoms.graph_dot(A, keys['x'], keys['Ax'])
#     #     subdasks['r'] = vector.graph_binaryop(operator.sub, b.name, keys['Ax'], keys['r'])

#     # if M is None:
#     #     subdasks['p'] = vector.dask_copy(subdasks['r'], keys['r'], keys['p'])
#     # else:
#     #     subdasks['p'] = atoms.graph_dot(M, keys['r'], keys['p'])

#     # subdasks['gamma'] = vector.graph_inner_product(keys['p'], keys['r'], keys['gamma'])
#     # subdasks['resnrm2'] = vector.graph_inner_product(keys['r'], keys['r'], keys['resnrm2'])

#     # dsk = dask.sharedict.merge(A.dask, b.dask)
#     # for sym in symbols:
#     #     dsk.update_with_key(subdasks[sym], keys[sym])
#     # tokens = {sym: keys[sym] for sym in ('x', 'r', 'p', 'gamma', 'resnrm2')}
#     # return block_cg_update_state(dsk, A, tokens, **options)



# def block_cg_initialize(A, b, M, x0, name=None, **options):
#     token = name or dask.base.tokenize(A)
#     itertoken = 'cg-iter-0-' + token

#     _r = 'r-' + itertoken
#     _x = 'x-' + itertoken
#     _Ax = 'Ax-' + itertoken
#     _p = 'p-' + itertoken
#     _gamma = 'gamma-' + itertoken
#     _resnrm2 = 'resnrm2-' + itertoken

#     options['use_graphdot'] = options.pop('use_graphdot', USE_GRAPHDOT)

#     if x0 is None:
#         dsk_r = da.core.top(lambda bi: bi, _r, 'i', b.name, 'i',
#                             numblocks={b.name: b.numblocks})
#         dsk_x = da.core.top(lambda ri: 0 * ri, _x, 'i', _r, 'i',
#                             numblocks={_r: b.numblocks})
#     else:
#         dsk_x = da.core.top(lambda x0i: x0i, _x, 'i', x0.name, 'i',
#                             numblocks={x0.name: b.numblocks})
#         if options['use_graphdot']:
#             dsk_Ax = atoms.graph_dot(A, _x, _Ax)
#         else:
#             dsk_Ax = da.core.top(da.core.dotmany, _Ax, 'i', A.name, 'ij', _x, 'j',
#                                  numblocks={A.name: A.numblocks, _x: b.numblocks})
#         dsk_r = da.core.top(operator.sub, _r, 'i', b.name, 'i', _Ax, 'i',
#                             numblocks={b.name: b.numblocks, _Ax: b.numblocks})

#     if M is None:
#         dsk_p = {(_p, key[1]): dsk_r[_r, key[1]] for key in dsk_r}
#     else:
#         dsk_p = atoms.graph_dot(M, _r, _p)

#     dsk_gamma = da.core.top(da.core.dotmany, _gamma, '', _r, 'i', _p, 'i',
#                               numblocks={_r: b.numblocks, _p: b.numblocks})
#     dsk_resnrm2 = da.core.top(da.core.dotmany, _resnrm2, '', _r, 'i', _r, 'i',
#                               numblocks={_r: b.numblocks})

#     dsk = dask.sharedict.merge(A.dask, b.dask)
#     dsk.update_with_key(dsk_x, _x)
#     if dsk_Ax is not None:
#         dsk.update_with_key(dsk_Ax, _Ax)
#     dsk.update_with_key(dsk_r, _r)
#     dsk.update_with_key(dsk_Mr, _Mr)
#     dsk.update_with_key(dsk_p, _p)
#     dsk.update_with_key(dsk_resnrm2, _resnrm2)
#     tokens = dict(x=_x, r=_r, p=_p, gamma=_gamma, resnrm2=_resnrm2)
#     return block_cg_update_state(dsk, A, tokens, **options)

# def block_cg_iterate(dsk, A, M, iteration, name=None, **options):
#     m, _ = A.shape
#     chunks_1d = (A.chunks[1],)
#     nblks_1d = (A.numblocks[0],)

#     def graph_inner_product(left, right, output_key):
#         return da.core.top(da.core.dotmany(
#                 output_key, '', left, 'i', right, 'i',
#                 numblocks={left: nblks_1d, right: nblks_1d}))

#     token = name or dask.base.tokenize(A)
#     itertoken = 'cg-iter-' + str(iteration) + '-' + token
#     oitertoken = 'cg-iter-' + str(iteration - 1) + '-' + token

#     _Ap = 'Ap-' + itertoken
#     _pAp = 'pAp-' + itertoken
#     _alpha = 'alpha-' + itertoken
#     _beta = 'beta-' + itertoken
#     _gamma = 'gamma-' + itertoken
#     _ogamma = 'gamma-' + oitertoken
#     _x = 'x-' + itertoken
#     _ox = 'x-' + oitertoken
#     _r = 'r-' + itertoken
#     _or = 'r-' + oitertoken
#     _p = 'p-' + itertoken
#     _op = 'p-' + oitertoken
#     _Mr = 'Mr-' + itertoken
#     _resnrm2 = 'resnrm2-' + itertoken

#     options['use_graphdot'] = options.pop('use_graphdot', USE_GRAPHDOT)

#     # alpha = oresnrm2 / p.dot(Ap)
#     if options['use_graphdot']:
#         dsk_Ap = atoms.graph_dot(A, _op, _Ap)
#     else:
#         dsk_Ap = da.core.top(da.core.dotmany, _Ap, 'i', A.name, 'ij', _op, 'j',
#                              numblocks={A.name: A.numblocks, _op: nblks_1d})

#     dsk_pAp = da.core.top(da.core.dotmany, _pAp, '', _op, 'i', _Ap, 'i',
#                             numblocks={_op: nblks_1d, _Ap: nblks_1d})
#     dsk_alpha = da.core.top(operator.div, _alpha, '', _ogamma, '', _pAp, '',
#                             numblocks={_ogamma: (), _pAp: ()})

#     # x = ox + alpha * p
#     def update_x(xi, pi, alpha): return xi + alpha * pi
#     dsk_x = da.core.top(update_x, _x, 'i', _ox, 'i', _op, 'i', _alpha, '',
#                         numblocks={_ox: nblks_1d, _op: nblks_1d, _alpha: ()})

#     # r = or - alpha * Ap
#     def update_r(ri, Api, alpha): return ri - alpha * Api
#     dsk_r = da.core.top(update_r, _r, 'i', _or, 'i', _Ap, 'i', _alpha, '',
#                         numblocks={_or: nblks_1d, _op: nblks_1d, _alpha: ()})

#     # gamma = r'Mr, resnrm2 = r'r
#     if M is None:
#         dsk_Mr = {(_Mr, rkey[1]): dsk_r[_r, rkey[1]] for rkey in dsk_r}
#     else:
#         if options['use_graphdot']:
#             dsk_Mr = atoms.graph_dot(M, _r, _Mr)
#         else:
#             raise NotImplementedError

#     # resnrm2 = r'Mr
#     dsk_gamma = da.core.top(da.core.dotmany, _resnrm2, '', _r, 'i', _Mr, 'i',
#                               numblocks={_r: nblks_1d, _Mr: nblks_1d})
#     dsk_resnrm2 = da.core.top(da.core.dotmany, _resnrm2, '', _r, 'i', _r, 'i',
#                               numblocks={_r: nblks_1d})

#     # p = Mr + (gamma / ogamma) * op
#     dsk_beta = da.core.top(operator.div, _beta, '', _gamma, '', _ogamma, '',
#                            numblocks={_gamma: (), _ogamma: ()})
#     def update_p(Mri, pi, beta): return Mri + beta * pi
#     dsk_p = da.core.top(update_p, _p, 'i', _Mr, 'i', _op, 'i', _beta, '',
#                           numblocks={_Mr: nblks_1d, _op: nblks_1d, _beta: ()})


#     dsk = dask.sharedict.merge(dsk, A.dask)
#     dsk.update_with_key(dsk_Ap, key=_Ap)
#     dsk.update_with_key(dsk_pAp, key=_pAp)
#     dsk.update_with_key(dsk_alpha, key=_alpha)
#     dsk.update_with_key(dsk_x, key=_x)
#     dsk.update_with_key(dsk_r, key=_r)
#     dsk.update_with_key(dsk_Mr, key=_Mr)
#     dsk.update_with_key(dsk_gamma, key=_gamma)
#     dsk.update_with_key(dsk_resnrm2, key=_resnrm2)
#     dsk.update_with_key(dsk_beta, key=_beta)
#     dsk.update_with_key(dsk_p, key=_p)
#     tokens = dict(x=_x, r=_r, p=_p, gamma=_gamma, resnrm2=_resnrm2)
#     return block_cg_update_state(dsk, A, tokens, **options)

# @dispatch(da.Array, da.Array)
# def cg(A, b, tol=1e-8, maxiters=500, verbose=0, x0=None,
#        preconditioner=None, name=None, **options):
#     """ Block conjugate gradient

#         Parameters
#         ----------
#         A: Dask.array.Array
#         b: Dask.array.Array
#         tol: float
#         maxiters: int
#         x0: optional, Dask.array.Array
#         preconditioner: options, Dask.array.Array

#         Returns:
#         --------
#         x: Dask.array.Array
#         iters: int
#         resnorm: float


#         Find x such that

#             Ax = b

#         for square, symmetric, block matrix A.

#         If a preconditioner M is provided, solve the left-preconditioned
#         equivalent problem,

#             M(Ax - b) = 0

#         N.B.: the blocks of A, M, b, and x0 must all be compatible
#     """
#     print_iter = max(1, maxiters / 10**verbose)

#     A, b, M = dask.persist(A, b, preconditioner)
#     block_cg_verify_dims(A, b, M, x0)

#     dsk, x, resnrm2 = block_cg_initialize(A, b, M, x0, name=name)
#     (resnrm2,) = dask.compute(resnrm2)
#     if resnrm2**0.5 < tol:
#         return x, 0, resnrm2**0.5

#     for k in range(1, maxiter + 1):
#         dsk, x, resnrm2 = block_cg_iterate(dsk, A, M, k, name=name)
#         (resnrm2,) = dask.compute(resnrm2)
#         if resnrm2**0.5 < tol:
#             break
#         elif k % print_iter == 0:
#             print('ITER: {:5}\t||Ax - b||_t'.format(k, resnrm2**0.5))

#     return x, k, resnrm2**0.5

# @dispatch(np.ndarray, np.ndarray)
# def cg(A, b, tol=1e-8, maxiters=500, verbose=0, x0=None, preconditioner=None,
#        **options):
#     """ Conjugate gradient

#         Parameters
#         ----------
#         A: array-like
#         b: array-like
#         tol: float
#         maxiters: int
#         x0: optional, array-like,
#         preconditioner: optional, array-like

#         Returns
#         -------
#         x: array-like
#         iters: int
#         resnorm: float
#     """
#     print_iter = max(1, maxiters / 10**verbose)

#     x = 0 * b if x0 is None else x0
#     r = b - A.dot(x)

#     M = preconditioner
#     Mr = r if M is None else M.dot(r)
#     p = Mr
#     norm_r = np.linalg.norm(r)
#     gamma = r.dot(Mr)

#     if norm_r < tol:
#         return x, 0, res

#     for k in range(1, maxiters + 1):
#         Ap = A.dot(p)
#         alpha = gamma / p.dot(Ap)
#         x += alpha * p
#         r -= alpha * Ap
#         Mr = r if M is None else M.dot(r)
#         norm_r = np.linalg.norm(r)

#         if norm_r < tol:
#             break
#         elif (k) % print_iter == 0:
#             print("ITER: {:5}\t||Ax -  b||_2: {}".format(k, norm_r))

#         gamma_next = r.dot(Mr)
#         p = Mr + (gamma_next / gamma) * p
#         gamma = gamma_next

#     return x, k, norm_r