# Test driver for mmio cosim.

import esi_cosim
import capnp
import sys

c = esi_cosim.LowLevel(sys.argv[3], sys.argv[2])
r = c.low.readMMIO(40).wait()
print(f"data resp: 0x{r.data:x}")

try:
  c.low.readMMIO(0).wait()
  assert False, "above should have thrown exception"
except capnp.lib.capnp.KjException:
  pass

c.low.writeMMIO(32, 86).wait()
r = c.low.readMMIO(32).wait()
print(f"data resp: 0x{r.data:x}")
assert r.data == 86

try:
  c.low.writeMMIO(0, 86).wait()
  assert False, "above should have thrown exception"
except capnp.lib.capnp.KjException:
  pass
