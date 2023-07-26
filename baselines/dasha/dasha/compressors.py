import numpy as np


def decompress(compressed_vector, assert_compressor=None):
    dim, indices, values, name = compressed_vector
    if assert_compressor is not None:
        assert assert_compressor == name
    decompressed_array = np.zeros((dim,), dtype=values.dtype)
    decompressed_array[indices] = values
    return decompressed_array


class BaseCompressor(object):
    def compress(self, vector):
        compressed_vector = self.compress_impl(vector)
        class_name = self.name()
        compressed_vector.append(class_name)
        return compressed_vector
    
    @classmethod
    def name(cls):
        return cls.__class__.__name__

    def compress_impl(self, vector):
        raise NotImplementedError()
    
    def num_nonzero_components(self):
        raise NotImplementedError()


class UnbiasedBaseCompressor(BaseCompressor):
    def omega(self):
        raise NotImplementedError()


class IdentityUnbiasedCompressor(UnbiasedBaseCompressor):
    def __init__(self, dim=None):
        self._dim = dim
    
    def compress_impl(self, vector):
        dim = vector.shape[0]
        return [np.array(dim, dtype=np.int32), np.arange(dim), np.copy(vector)]
    
    def omega(self):
        return 0
    
    def num_nonzero_components(self):
        return self._dim
