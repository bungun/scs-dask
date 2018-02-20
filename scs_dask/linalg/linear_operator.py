import abc
import six
import operator
import dask
import dask.array as da
import numpy as np

@six.add_metaclass(abc.ABCMeta)
class DaskLinearOperator(object):
    def __init__(self, name, shape, chunks, dtype):
        self.name = name
        self.shape = shape
        self.chunks = chunks
        self.nblocks = tuple(len(cc) for cc in array.chunks)
        self.dtype = dtype
        self.dask = dask.sharedict.ShareDict()

    @abc.abstractmethod
    def graph_apply(self, dsk, input_key, output_key, **options):
        pass


@six.add_metaclass(abc.ABCMeta)
class DaskLinearOperator(dask.base.DaskMethodsMixin):
    def __init__(self, dsk, shape, chunks, dtype, opname='linear-operator'):
        self._dsk = dsk
        self._keys = list()
        self._opname = str(opname)
        self.shape = shape
        self.chunks = chunks
        self.nblocks = tuple(len(cc) for cc in chunks)
        self.dtype = dtype

    def __dask_graph__(self):
        return self._dsk

    def __dask_keys__(self):
        return self._keys

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    # Use the threaded scheduler by default.
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def __dask_postcompute__(self):
        # We want to return the results as a tuple, so our finalize
        # function is `tuple`. There are no extra arguments, so we also
        # return an empty tuple.
        return dask.array.Array, ()

    def __dask_postpersist__(self):
        return DaskLinearOperator, (self._keys,)

    def __dask_tokenize__(self):
        return (self._opname,) + tuple(self._keys)

    @abc.abstractmethod
    def graph_apply(self, dsk, input_key, output_key, **options):
        pass


@six.add_metaclass(abc.ABCMeta)
class DLOSymmetric(DaskLinearOperator):
    def __init__(self, name, shape, chunks, dtype):
        assert shape[0] == shape[0]
        assert chunks[0] == chunks[1]
        DaskLinearOperator.__init__(self, name, shape, chunks, dtype)

class DLODense(DaskLinearOperator):
    def __init__(self, array, name=None):
        assert isinstance(array, da.Array), 'input is dask.array.Array'
        self.data = array
        self.dask = dask.sharedict.merge(array.dask)
        name = array.name if name is None else name
        name = 'dense-operator-' + name
        DaskLinearOperator.__init__(
                self, name, array.shape, array.chunks, array.dtype)

    def graph_apply(self, dsk, input_key, output_key, transpose=False,
                    **options):
        if not transpose:
            idx_out, idx_arr, idx_in = 'i', 'ij', 'j'
            blk_arr, blk_in = self.nblocks, (self.nblocks[0],)
            fcn = None
        else:
            idx_out, idx_arr, idx_in = 'j', 'ji', 'i'
            blk_arr, blk_in = self.nblocks[::-1], (self.nblocks[1],)
            fcn = da.transpose

        dsk_out = da.core.top(
                da.core.dotmany,
                output_key, idx_out,
                self.data.name, idx_arr,
                input_key, idx_in,
                leftfunction=fcn,
                numblocks={self.data.name: blk_arr, input_key: blk_in})
        return dask.sharedict.merge(self.dask, (dsk_out, output_key))

class DLODiagonal(DLOSymmetric):
    def __init__(self, vector, name=None):
        assert isinstance(vector, da.Array), 'input is dask.array.Array'
        self.data = vector
        self.dask = dask.sharedict.merge(vector.dask)
        name = vector.name if name is None else name
        name = 'diagonal-operator-' + name
        shape = (vector.shape[0], vector.shape[0])
        chunks = (vector.chunks[0], vector.chunks[0])
        DLOSymmetric.__init__(self, name, shape, chunks, vector.dtype)

    def graph_apply(self, dsk, input_key, output_key, **options):
        nblks = (self.nblocks[1],)
        dsk_out = da.core.top(
                operator.mul,
                output_key, 'i',
                self.data.name, 'i',
                input_key, 'i',
                numblocks={self.data.name: nblks, input_key: nblks})
        return dask.sharedict.merge(self.dask, (dsk_out, output_key))

class DLOGram(DLOSymmetric):
    def __init__(self, dsk_operator, name=None, **options):
        assert isinstance(dsk_operator, DaskLinearOperator), 'input is DaskLinearOperator'
        self.op = dsk_operator
        self.transpose = dsk_operator.shape[0] < dsk_operator.shape[1]
        name = self.op.name if name is None else name
        name = 'gram-operator-' + self.op.name
        mindim = min(dsk_operator.shape)
        shape = (mindim, mindim)
        chunks = (dsk_operator.chunks[0], dsk_operator.chunks[1])
        if self.transpose:
            chunks = chunks[::-1]
        DaskSymmetricOperator.__init__(self, name, shape, chunks, self.op.dtype)

    def graph_apply(self, dsk, input_key, output_key, **options):
        if not self.transpose:
            # build A'Ax
            step_key = name + '-apply-' + input_key
            dsk = self.op.graph_apply(dsk, input_key, step_key)
            return self.op.graph_apply(dsk, step_key, output_key, tranpose=True)
        else:
            # build AA'x
            steo_key = name + '-adjoint-' + input_key
            dsk = self.op.graph_apply(dsk, input_key, step_key, transpose=True)
            return self.op.graph_apply(dsk, step_key, output_key)

class DLORegularizedGram(DaskLinearOperator):
    def __init__(self, dsk_operator, regularization=1):
        if not isinstance(dsk_operator, DaskGram):
            dsk_operator = DLOGram(dsk_operator)
        self.op = dsk_operator
        self.regularization = float(regularization)
        name = self.op.name if name is None else name
        name = 'regularized-' + self.op.name
        DLOSymmetric.__init__(
                self, name, self.op.shape, self.op.chunks, self.op.dtype)

    def graph_apply(self, dsk, input_key, output_key, **options):
        dsk = self.op.graph_apply(dsk, input_key, step_key)
        def regularize(x, AAx):
            return self.regularization * x + AAx
        nblks = (self.op.nblocks[1],)
        dsk_IAA = da.core.top(
                regularize,
                output_key, 'i',
                input_key, 'i',
                step_key, 'i',
                numblocks={input_key: nblks, step_key: nblks})
        return dask.sharedict.merge(dsk, (dsk_IAA, output_key))



