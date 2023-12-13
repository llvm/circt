import esi
import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1
print(m.type_table)

d = m.build_accelerator(acc)

recv = d.ports[esi.AppID("loopback_inout")].channels["resp"]
recv.connect()

send = d.ports[esi.AppID("loopback_inout")].channels["req"]
send.connect()

data = [24, 42, 36]
send.write(data)
resp = []
# Reads are non-blocking, so we need to poll.
while resp == []:
  resp = recv.read(2)

print(f"data: {data}")
print(f"resp: {resp}")
assert resp[0] == data[0] + 7
assert resp[1] == data[1]

print("PASS")
