import esiaccel as esi

import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

mmio = acc.get_service_mmio()
data = mmio.read(8)
assert data == 0x207D98E5E5100E51

################################################################################
# MMIOClient tests
################################################################################


def read_offset(mmio_offset: int, offset: int, add_amt: int):
  data = mmio.read(mmio_offset + offset)
  if data == add_amt + offset:
    print(f"PASS: read_offset({mmio_offset}, {offset}, {add_amt}) -> {data}")
  else:
    assert False, f"read_offset({mmio_offset}, {offset}, {add_amt}) -> {data}"


# MMIO offset into mmio_client[9]. TODO: get this from the manifest. API coming.
mmio_client_9_offset = 131072
read_offset(mmio_client_9_offset, 0, 9)
read_offset(mmio_client_9_offset, 13, 9)

# MMIO offset into mmio_client[4].
mmio_client_4_offset = 65536
read_offset(mmio_client_4_offset, 0, 4)
read_offset(mmio_client_4_offset, 13, 4)

# MMIO offset into mmio_client[14].
mmio_client_14_offset = 196608
read_offset(mmio_client_14_offset, 0, 14)
read_offset(mmio_client_14_offset, 13, 14)

################################################################################
# Manifest tests
################################################################################

assert acc.sysinfo().esi_version() == 0
m = acc.manifest()
assert m.api_version == 0
print(m.type_table)

d = acc.build_accelerator()

recv = d.ports[esi.AppID("loopback_add7")].read_port("result")
recv.connect()

send = d.ports[esi.AppID("loopback_add7")].write_port("arg")
send.connect()

################################################################################
# Loopback add 7 tests
################################################################################

data = 10234
send.write(data)
got_data = False
resp = recv.read()

print(f"data: {data}")
print(f"resp: {resp}")
assert resp == data + 7

print("PASS")
