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


class RandKCompressor(UnbiasedBaseCompressor):
    def __init__(self, number_of_coordinates, seed, dim=None):
        self._number_of_coordinates = number_of_coordinates
        self._dim = dim
        self._generator = np.random.default_rng(seed)
    
    def num_nonzero_components(self):
        return self._number_of_coordinates

    def compress_impl(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates >= 0
        indices = self._generator.choice(dim, self._number_of_coordinates, replace = False)
        values = vector[indices] * float(dim / self._number_of_coordinates)
        return [np.array(dim, dtype=np.int32), indices, values]
    
    def omega(self):
        assert self._dim is not None
        return float(self._dim) / self._number_of_coordinates - 1
