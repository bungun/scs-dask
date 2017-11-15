"""Solve a linear system with conjugate gradient."""

import dask
# import dask.array as da


def cg(A, b, tol=1e-8, maxiters=500, verbose=0, x0=None, preconditioner=None):
    """ Conjugate gradient

        Parameters
        ----------
        A: array-like
        b: array-like
        tol: float
        maxiters: int
        x0: optional, array-like,
        preconditioner: optional, array-like
        client: optional, dask.distributed.Client

        Returns
        -------
        x: array-like
        iters: int
        resnorm: float

        Find x such that

            Ax = b

        for square, symmetric A.

        If a preconditioner M is provided, solve the left-preconditioned
        equivalent problem,

            M(Ax - b) = 0
    """
    print_iter = max(1, maxiter / 10**verbose)

    A, b, M = dask.persist(A, b, preconditioner)

    if x0 is None:
        r = 1 * b
        x = 0 * b
    else:
        r = 1 * b - A.dot(x0)
        x = x0

    Mr = r if M is None else M.dot(r)
    p = Mr
    resnrm2 = r.dot(Mr)

    x, r, p, resnrm2 = dask.persist(x, r, p, resnrm2)
    (resnrm2,) = dask.compute(resnrm2)
    if resnrm2**0.5 < tol:
        return x, 0, resnrm2**0.5

    for k in range(maxiter):
        ox, ores, op, oresnrm2 = x, r, p, resnrm2

        Ap = A.dot(p)
        alpha = resnrm2 / p.dot(Ap)
        x = ox + alpha * p
        r = ores - alpha * Ap
        Mr = r if M is None else M.dot(r)
        resnrm2 = r.dot(Mr)

        x, r, resnrm2 = dask.persist(x, r, resnrm2)
        (resnrm2,) = dask.compute(resnrm2)

        if resnrm2**0.5 < tol:
            break
        elif (k + 1) % print_iter == 0:
            print("ITER: {:5}\t||Ax -  b||_2: {}".format(k + 1, resnrm2**0.5))

        p = Mr + (resnrm2 / oresnrm2) * op
        x, r, resnrm2, p= dask.persist(x, r, resnrm2, p)

        (p,) = dask.persist(p)

    return x, k + 1, resnrm2**0.5
