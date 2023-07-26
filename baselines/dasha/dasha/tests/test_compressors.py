import unittest

import numpy as np

from dasha.compressors import decompress, IdentityUnbiasedCompressor


class TestIdentityUnbiasedCompressor(unittest.TestCase):

    def test(self) -> None:
        vec = np.random.rand(10)
        compressed_vec = IdentityUnbiasedCompressor().compress(vec)
        np.testing.assert_almost_equal(vec, decompress(compressed_vec, assert_compressor=IdentityUnbiasedCompressor.name()))


if __name__ == "__main__":
    unittest.main()
