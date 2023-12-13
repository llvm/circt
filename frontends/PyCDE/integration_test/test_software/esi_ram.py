from typing import List
import esi
import random
import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

m = acc.manifest()
d = m.build_accelerator(acc)

mem_write = d.ports[esi.AppID("write")].channels["write"]
mem_write.connect()
mem_read_addr = d.ports[esi.AppID("read")].channels["address"]
mem_read_addr.connect()
mem_read_data = d.ports[esi.AppID("read")].channels["data"]
mem_read_data.connect()


def read(addr: int) -> List[int]:
  mem_read_addr.write([addr])
  resp: List[int] = []
  while resp == []:
    resp = mem_read_data.read(8)
  print(f"resp: {resp}")
  return resp


# The contents of address 3 are continuously updated to the contents of address
# 2 by the accelerator.
data = [random.randint(0, 2**8 - 1) for _ in range(8)]
mem_write.write(data + [2])
resp = read(2)
assert resp == data
resp = read(3)
assert resp == data

# Check this by writing to address 3 and reading from it. Shouldn't have
# changed.
zeros = [0] * 8
mem_write.write(zeros + [3])
resp = read(3)
assert resp == data
