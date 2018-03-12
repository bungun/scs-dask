""" Type hierarchy for abstract linear operators
"""
class DaskLinearOperator(object):
    """ Base class for abstract :obj:`dask.array.Array`-based linear operators
    """
    def __init__(self, data, name=None):
        self._data = data
        self._name = 'DLO-' + data.name if name is None else str(name)

    @property
    def data(self):
        return self._data

    @property
    def root_data(self):
        if isinstance(self._data, DaskLinearOperator):
            return self._data.root_data
        else:
            return self._data

    @property
    def dask(self):
        return self._data.dask

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def chunks(self):
        return self._data.chunks

    @property
    def numblocks(self):
        return self._data.numblocks

    @property
    def dtype(self):
        return self._data.dtype

    def __dask_tokenize__(self):
        return (type(self), self._data, self._name)

class DLOSymmetric(DaskLinearOperator):
    def __init__(self, data, name=None):
        name = 'DLO-symmetric-' + data.name if name is None else name
        DaskLinearOperator.__init__(self, data, name=name)
        assert self.shape[0] == self.shape[1]
        assert self.chunks[0] == self.chunks[1]

class DLODense(DaskLinearOperator):
    def __init__(self, data, name=None):
        name = 'DLO-dense-' + data.name if name is None else name
        DaskLinearOperator.__init__(self, data, name=name)

class DLODiagonal(DLOSymmetric):
    def __init__(self, data, name=None):
        name = 'DLO-diagonal-' + data.name if name is None else name
        DLOSymmetric.__init__(self, data, name=name)

    @property
    def shape(self):
        return self._data.shape[0], self._data.shape[0]

    @property
    def chunks(self):
        return self._data.chunks[0], self._data.chunks[0]

    @property
    def numblocks(self):
        return self._data.numblocks[0], self._data.numblocks[0]

class DLOGram(DLOSymmetric):
    def __init__(self, data, name=None, **options):
        self.transpose = options.pop('transpose', data.shape[0] < data.shape[1])
        self._idx = 1 - int(self.transpose)
        name = 'DLO-gram-' + data.name if name is None else name
        DLOSymmetric.__init__(self, data, name=name)

    @property
    def shape(self):
        return self._data.shape[self._idx], self._data.shape[self._idx]

    @property
    def chunks(self):
        return self._data.chunks[self._idx], self._data.chunks[self._idx]

    @property
    def numblocks(self):
        return self._data.numblocks[self._idx], self._data.numblocks[self._idx]

    def __dask_tokenize__(self):
        return (type(self), self._data, self._name, self.transpose)

class DLORegularizedGram(DLOGram):
    def __init__(self, data, name=None, regularization=1., **options):
        self.regularization = float(regularization)
        name = 'DLO-regularized-gram-' + data.name if name is None else name
        DLOGram.__init__(self, data, name=name, **options)


