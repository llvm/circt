from typing import List
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

mysvc_send = loopback.ports[esi.AppID("mysvc_recv")].channels["recv"]
mysvc_send.connect()
mysvc_send.write([0])

mysvc_send = loopback.ports[esi.AppID("mysvc_send")].channels["send"]
mysvc_send.connect()
resp: List[int] = []
# Reads are non-blocking, so we need to poll.
while resp == []:
  print("i0 polling")
  resp = mysvc_send.read(1)
print(f"i0 resp: {resp}")

recv = loopback.ports[esi.AppID("loopback_tohw")].channels["recv"]
recv.connect()

send = loopback.ports[esi.AppID("loopback_fromhw")].channels["send"]
send.connect()

data = [24]
recv.write(data)
resp = []
# Reads are non-blocking, so we need to poll.
while resp == []:
  print("polling")
  resp = send.read(1)

# Trace platform intentionally produces random responses.
if platform != "trace":
  print(f"data: {data}")
  print(f"resp: {resp}")
  assert resp == data

# Placeholder until we have a runtime function API.
myfunc = d.ports[esi.AppID("func1")]
myfunc.channels["arg"].connect()
myfunc.channels["result"].connect()

print("PASS")
