import esiaccel
import IPython

conn = esiaccel.AcceleratorConnection("cosim", "env")
acc = conn.build_accelerator()
rdaddr = acc.ports[esiaccel.AppID("mmio_axi_rdaddr")].write_port("data")
rdaddr.connect()
rddata = acc.ports[esiaccel.AppID("mmio_axi_rddata")].read_port("data")
rddata.connect()

for addr in range(0, 0x100, 4):
  rdaddr.write(addr)
  d = rddata.read()
  print(f"Read from 0x{addr:02x}: 0x{d:08x}")

for addr in range(0x00010000, 0x00010100, 4):
  rdaddr.write(addr)
  d = rddata.read()
  print(f"Read from 0x{addr:02x}: 0x{d:08x}")

# IPython.embed()
