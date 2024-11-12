import esiaccel
import IPython

conn = esiaccel.AcceleratorConnection("cosim", "env")
acc = conn.build_accelerator()
rdaddr = acc.ports[esiaccel.AppID("mmio_axi_rdaddr")].write_port("data")
rdaddr.connect()
rddata = acc.ports[esiaccel.AppID("mmio_axi_rddata")].read_port("data")
rddata.connect()
print("connected")


def read(addr):
  rdaddr.write(addr)
  d = rddata.read()
  print(f"Read from 0x{addr:02x}: 0x{d:08x}")
  return d


# for addr in range(0, 0x100, 4):
#   rdaddr.write(addr)
#   d = rddata.read()
#   # print(f"Read from 0x{addr:02x}: 0x{d:08x}")

# # for addr in range(0x00010000, 0x00010100, 4):
# #   rdaddr.write(addr)
# #   d = rddata.read()
# #   print(f"Read from 0x{addr:02x}: 0x{d:08x}")

wraddr = acc.ports[esiaccel.AppID("mmio_axi_wraddr")].write_port("data")
wraddr.connect()
wrdata = acc.ports[esiaccel.AppID("mmio_axi_wrdata")].write_port("data")
wrdata.connect()
wrresp = acc.ports[esiaccel.AppID("mmio_axi_wrresp")].read_port("data")
wrresp.connect()


def write(addr, data):
  wraddr.write(addr)
  wrdata.write(data)
  d = wrresp.read()
  print(f"Write to 0x{addr:02x}: 0x{data:08x} -> {d}")


write(262144 + 8, 30)
write(262144 + 12, 40)
print(read(262144 + 32))
print(read(262144 + 36))

# IPython.embed()
