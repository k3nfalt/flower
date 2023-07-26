import numpy as np


class BaseCompressor(object):
    def compress(self, vector):
        raise NotImplementedError()

    def num_nonzero_components(self):
        raise NotImplementedError()


class UnbiasedBaseCompressor(BaseCompressor):
    def omega(self):
        raise NotImplementedError()


def decompress(dim, indices, values):
    decompressed_array = np.zeros((dim,), dtype=values.dtype)
    decompressed_array[indices] = values
    return decompressed_array


class IdentityUnbiasedCompressor(UnbiasedBaseCompressor):
    def __init__(self, dim=None):
        self._dim = dim
    
    def compress(self, vector):
        dim = vector.shape[0]
        return [np.array(dim, dtype=np.int32), np.arange(dim), np.copy(vector)]
    
    def omega(self):
        return 0
    
    def num_nonzero_components(self):
        return self._dim
