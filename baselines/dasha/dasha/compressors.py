import numpy as np


def decompress(compressed_vector, assert_compressor=None):
    dim, indices, values, name = compressed_vector
    if assert_compressor is not None:
        assert assert_compressor == name
    decompressed_array = np.zeros((dim,), dtype=values.dtype)
    decompressed_array[indices] = values
    return decompressed_array


BITS_IN_FLOAT_32 = 32


def estimate_size(compressed_vector):
    _, indices, values, _ = compressed_vector
    assert values.dtype == np.float32
    return (len(indices) + len(values)) * BITS_IN_FLOAT_32


class BaseCompressor(object):
    def __init__(self, seed, dim) -> None:
        self._seed = seed
        self._dim = dim
    
    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        compressed_vector = self.compress_impl(vector)
        class_name = self.name()
        compressed_vector.append(class_name)
        return compressed_vector
    
    @classmethod
    def name(cls):
        return cls.__name__
    
    def set_dim(self, dim):
        assert self._dim is None or self._dim == dim
        self._dim = dim

    def compress_impl(self, vector):
        raise NotImplementedError()
    
    def num_nonzero_components(self):
        raise NotImplementedError()


class UnbiasedBaseCompressor(BaseCompressor):
    def omega(self):
        raise NotImplementedError()


class IdentityUnbiasedCompressor(UnbiasedBaseCompressor):
    def __init__(self, seed=None, dim=None):
        super(IdentityUnbiasedCompressor, self).__init__(seed=seed, dim=dim)
    
    def compress_impl(self, vector):
        dim = self._dim
        return [np.array(dim, dtype=np.int32), np.arange(dim), np.copy(vector)]
    
    def omega(self):
        return 0
    
    def num_nonzero_components(self):
        assert self._dim is not None
        return self._dim


class RandKCompressor(UnbiasedBaseCompressor):
    def __init__(self, seed, number_of_coordinates, dim=None):
        super(RandKCompressor, self).__init__(seed=seed, dim=dim)
        self._number_of_coordinates = number_of_coordinates
        self._generator = np.random.default_rng(seed)
    
    def num_nonzero_components(self):
        return self._number_of_coordinates

    def compress_impl(self, vector):
        assert self._number_of_coordinates >= 0
        dim = self._dim
        indices = self._generator.choice(dim, self._number_of_coordinates, replace = False)
        values = vector[indices] * float(dim / self._number_of_coordinates)
        return [np.array(dim, dtype=np.int32), indices, values]
    
    def omega(self):
        assert self._dim is not None
        return float(self._dim) / self._number_of_coordinates - 1
