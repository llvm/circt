from telnetlib import IP
from typing import Optional
import esi
import random
import sys

import IPython

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

# IPython.embed()

d = acc.build_accelerator()

mem_write = d.ports[esi.AppID("write")].write_port("req")
mem_write.connect()
mem_read_addr = d.ports[esi.AppID("read")].write_port("address")
mem_read_addr.connect()
mem_read_data = d.ports[esi.AppID("read")].read_port("data")
mem_read_data.connect()


def read(addr: int) -> bytearray:
  mem_read_addr.write([addr])
  got_data = False
  resp: Optional[bytearray] = None
  while not got_data:
    (got_data, resp) = mem_read_data.read()
  print(f"resp: {resp}")
  return resp


# The contents of address 3 are continuously updated to the contents of address
# 2 by the accelerator.
data = bytearray([random.randint(0, 2**8 - 1) for _ in range(8)])
mem_write.write({"address": [2], "data": data})
resp = read(2)
assert resp == data
resp = read(3)
assert resp == data

# Check this by writing to address 3 and reading from it. Shouldn't have
# changed.
zeros = bytearray([0] * 8)
mem_write.write({"address": [3], "data": zeros})
resp = read(3)
assert resp == data

mmio = acc.get_service_mmio()
for addr in range(0, 16, 4):
  print(f"MMIO {addr}: {mmio.read(addr)}")
