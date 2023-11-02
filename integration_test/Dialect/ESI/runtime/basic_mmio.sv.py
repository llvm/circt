import esi
import os
import sys

conn = f"{sys.argv[1]}:{sys.argv[2]}"

acc = esi.Accelerator("cosim", conn)
mmio = acc.get_service_mmio()

r = mmio.read(40)
print(f"data resp: 0x{r:x}")

try:
  mmio.read(0)
  assert False, "above should have thrown exception"
except Exception:
  print("caught expected exception")

mmio.write(32, 86)
r = mmio.read(32)
print(f"data resp: 0x{r:x}")
assert r == 86

try:
  mmio.write(0, 44)
  assert False, "above should have thrown exception"
except Exception:
  print("caught expected exception")
