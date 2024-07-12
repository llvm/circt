import esiaccel as esi

import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

mmio = acc.get_service_mmio()
data = mmio.read(8)
print(f"mmio data@8: {data:X}")
assert data == 0x207D98E5E5100E51

assert acc.sysinfo().esi_version() == 0
m = acc.manifest()
assert m.api_version == 0
print(m.type_table)

d = acc.build_accelerator()

recv = d.ports[esi.AppID("loopback_add7")].read_port("result")
recv.connect()

send = d.ports[esi.AppID("loopback_add7")].write_port("arg")
send.connect()

data = 10234
send.write(data)
got_data = False
resp = recv.read()

print(f"data: {data}")
print(f"resp: {resp}")
assert resp == data + 7

print("PASS")
