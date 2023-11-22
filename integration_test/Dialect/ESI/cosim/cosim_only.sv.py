# Test driver for mmio cosim.

import json
import sys
import zlib

import loopback as test

rpc = test.LoopbackTester(sys.argv[3], sys.argv[2])
print(rpc.cosim.list().wait())
rpc.test_list()
rpc.test_3bytes(5)

compressed_mani = rpc.cosim.getCompressedManifest().wait().compressedManifest
mani_str = zlib.decompress(compressed_mani).decode("ascii")
mani = json.loads(mani_str)
assert (mani["api_version"] == 1)
