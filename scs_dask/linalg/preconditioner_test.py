import dask.array as da

import scs_dask.linalg.linear_operator as linop
import scs_dask.linalg.preconditioner as pre

def is_inverse(preconditioner, diag):
    norm = da.linalg.norm((preconditioner.data * diag) - 1).compute()
    return norm < 1e-15 * (1 + diag.size**0.5)

def test_jacobi_preconditioner():
    A_test = da.random.random((100, 100), chunks=20)
    d_test = da.diag(A_test)
    dd_test = da.diag(A_test.T.dot(A_test))
    assert is_inverse(pre.jacobi_preconditioner(A_test), d_test)
    assert is_inverse(pre.jacobi_preconditioner(linop.DLODense(A_test)), d_test)
    assert is_inverse(pre.jacobi_preconditioner(linop.DLOGram(A_test)), dd_test)
    assert is_inverse(
            pre.jacobi_preconditioner(linop.DLORegularizedGram(A_test)),
            1 + dd_test)
    mu = da.random.normal(1, 1, (), chunks=())
    assert is_inverse(
            pre.jacobi_preconditioner(linop.DLORegularizedGram(A_test, regularization=mu)),
            mu + dd_test)
