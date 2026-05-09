# Script to help test xrt in cosim mode. No support for xrt in cosim mode in the
# runtime, so this is not an automated test. Just an ad-hoc one.

import esiaccel

conn = esiaccel.AcceleratorConnection("cosim", "env")
acc = conn.build_accelerator()
rdaddr = acc.ports[esiaccel.AppID("mmio_axi_rdaddr")].write_port("data")
rdaddr.connect()
rddata = acc.ports[esiaccel.AppID("mmio_axi_rddata")].read_port("data")
rddata.connect()

wraddr = acc.ports[esiaccel.AppID("mmio_axi_wraddr")].write_port("data")
wraddr.connect()
wrdata = acc.ports[esiaccel.AppID("mmio_axi_wrdata")].write_port("data")
wrdata.connect()
wrresp = acc.ports[esiaccel.AppID("mmio_axi_wrresp")].read_port("data")
wrresp.connect()

print("connected")


def read(addr):
  rdaddr.write(addr)
  d = rddata.read()
  print(f"Read from 0x{addr:02x}: 0x{d:08x}")
  return d


def write(addr, data):
  wraddr.write(addr)
  wrdata.write(data)
  d = wrresp.read()
  print(f"Write to 0x{addr:02x}: 0x{data:08x} -> {d}")


for addr in range(0, 0x100, 4):
  d = read(addr)
  if addr == 0x18:
    mani_offset = d

for addr in range(mani_offset, mani_offset + 0x100, 4):
  read(addr)

write(262144 + 8, 30)
write(262144 + 12, 40)
read(262144 + 32)
read(262144 + 36)
