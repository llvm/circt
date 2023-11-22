import esi
import sys

platform = sys.argv[1]
acc = esi.Accelerator(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1
print(m.type_table)

d = m.build_design(acc)

loopback = d.children[esi.AppID("loopback_inst", 0)]
appid = loopback.id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0

recv = loopback.ports[esi.AppID("loopback_tohw")].channels["recv"]
recv.connect()

send = loopback.ports[esi.AppID("loopback_fromhw")].channels["send"]
send.connect()

data = [24]
recv.write(data)
resp = []
# Reads are non-blocking, so we need to poll.
while resp == []:
  resp = send.read(1)

# Trace platform intentionally produces random responses.
if platform != "trace":
  print(f"data: {data}")
  print(f"resp: {resp}")
  assert resp == data

print("PASS")
