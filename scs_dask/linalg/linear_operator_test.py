import dask.array as da

import linear_operator as linop

def test_linear_operators():
    A = da.random.random((100, 50), chunks=20)
    Adlo = linop.DaskLinearOperator(A)
    assert Adlo.size == A.size
    assert Adlo.shape == A.shape
    assert Adlo.chunks == A.chunks
    assert Adlo.numblocks == A.numblocks
    assert Adlo.dtype == A.dtype

    try:
        linop.DLOSymmetric(A)
    except AssertionError:
        print 'fail on dims'

    try:
        linop.DLOSymmetric(da.random.random((100, 100), chunks=(10,20)))
    except AssertionError:
        print 'fail on chunks'

    Asymm = linop.DLOSymmetric(da.random.random((100, 100), chunks=10))
    assert Asymm.numblocks == (10, 10)

    Adn = linop.DLODense(A)
    assert Adn.numblocks == A.numblocks

    Adiag = linop.DLODiagonal(da.diag(da.random.random((100, 100), chunks=50)))
    assert Adiag.numblocks == (2, 2)
    assert Adiag.data.numblocks == (2,)

    Agm = linop.DLOGram(A)
    assert Agm.numblocks == (A.numblocks[1], A.numblocks[1])
    Agm2 = linop.DLOGram(da.transpose(A))
    assert Agm.shape == Agm2.shape

    Agm = linop.DLORegularizedGram(A)
    assert Agm.regularization == 1