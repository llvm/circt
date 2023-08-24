import esi_cosim
import capnp
import sys

c = esi_cosim.LowLevel(sys.argv[2], sys.argv[1])
r = c.low.readMMIO(40).wait()
print(f"data resp: 0x{r.data:x}")

try:
  c.low.readMMIO(0).wait()
  assert False, "above should have thrown exception"
except capnp.lib.capnp.KjException:
  pass
