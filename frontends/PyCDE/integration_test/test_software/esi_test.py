import esiaccel as esi
from esiaccel.types import MMIORegion

import sys
import time

esi.accelerator.ctxt.set_stdio_logger(esi.accelerator.cpp.LogLevel.Debug)

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

mmio = acc.get_service_mmio()
data = mmio.read(8)
assert data == 0x207D98E5E5100E51

assert acc.sysinfo().esi_version() == 0
m = acc.manifest()
assert m.api_version == 0
print(m.type_table)

d = acc.build_accelerator()

mmio_svc: esi.accelerator.MMIO
for svc in d.services:
  if isinstance(svc, esi.accelerator.MMIO):
    mmio_svc = svc
    break

for id, region in mmio_svc.regions.items():
  print(f"Region {id}: {region.base} - {region.base + region.size}")

assert len(mmio_svc.regions) == 4

################################################################################
# MMIOClient tests
################################################################################


def read_offset(mmio_x: MMIORegion, offset: int, add_amt: int):
  data = mmio_x.read(offset)
  if data == add_amt + offset:
    print(f"PASS: read_offset({offset}, {add_amt}) -> {data}")
  else:
    assert False, f"read_offset({offset}, {add_amt}) -> {data}"


mmio9 = d.ports[esi.AppID("mmio_client", 9)]
read_offset(mmio9, 0, 9)
read_offset(mmio9, 13, 9)

mmio4 = d.ports[esi.AppID("mmio_client", 4)]
read_offset(mmio4, 0, 4)
read_offset(mmio4, 13, 4)

mmio14 = d.ports[esi.AppID("mmio_client", 14)]
read_offset(mmio14, 0, 14)
read_offset(mmio14, 13, 14)

################################################################################
# MMIOReadWriteClient tests
################################################################################

mmio_rw = d.ports[esi.AppID("mmio_rw_client")]


def read_offset_check(i: int, add_amt: int):
  d = mmio_rw.read(i)
  if d == i + add_amt:
    print(f"PASS: read_offset_check({i}): {d}")
  else:
    assert False, f": read_offset_check({i}): {d}"


add_amt = 137
mmio_rw.write(8, add_amt)
read_offset_check(0, add_amt)
read_offset_check(12, add_amt)
read_offset_check(0x140, add_amt)

################################################################################
# Manifest tests
################################################################################

loopback = d.children[esi.AppID("loopback")]
recv = loopback.ports[esi.AppID("add")].read_port("result")
recv.connect()

send = loopback.ports[esi.AppID("add")].write_port("arg")
send.connect()

loopback_info = None
for mod_info in m.module_infos:
  if mod_info.name == "LoopbackInOutAdd":
    loopback_info = mod_info
    break
assert loopback_info is not None
add_amt = mod_info.constants["add_amt"].value

################################################################################
# Loopback add 7 tests
################################################################################

data = 10234
# Blocking write interface
send.write(data)
resp = recv.read()

print(f"data: {data}")
print(f"resp: {resp}")
assert resp == data + add_amt

# Non-blocking write interface
data = 10235
nb_wr_start = time.time()

# Timeout of 5 seconds
nb_timeout = nb_wr_start + 5
write_succeeded = False
while time.time() < nb_timeout:
  write_succeeded = send.try_write(data)
  if write_succeeded:
    break

assert write_succeeded, "Non-blocking write failed"
resp = recv.read()
print(f"data: {data}")
print(f"resp: {resp}")
assert resp == data + add_amt

print("PASS")

################################################################################
# Const producer tests
################################################################################

producer_bundle = d.ports[esi.AppID("const_producer")]
producer = producer_bundle.read_port("data")
producer.connect()
data = producer.read()
producer.disconnect()
print(f"data: {data}")
assert data == 42

################################################################################
# Handshake JoinAddFunc tests
################################################################################

# Disabled test since the DC dialect flow is broken. Leaving the code here in
# case someone fixes it.

# a = d.ports[esi.AppID("join_a")].write_port("data")
# a.connect()
# b = d.ports[esi.AppID("join_b")].write_port("data")
# b.connect()
# x = d.ports[esi.AppID("join_x")].read_port("data")
# x.connect()

# a.write(15)
# b.write(24)
# xdata = x.read()
# print(f"join: {xdata}")
# assert xdata == 15 + 24
