from typing import List, Optional
import esi
import esi.types as types
import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1

for esiType in m.type_table:
  print(f"{esiType}")

d = acc.build_accelerator()

loopback = d.children[esi.AppID("loopback_inst", 0)]
appid = loopback.id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0

mysvc_send = loopback.ports[esi.AppID("mysvc_recv")].write_port("recv")
mysvc_send.connect()
mysvc_send.write(None)
print(f"mysvc_send.type: {mysvc_send.type}")
assert isinstance(mysvc_send.type, types.VoidType)

mysvc_send = loopback.ports[esi.AppID("mysvc_send")].read_port("send")
mysvc_send.connect()
resp: bool = False
# Reads are non-blocking, so we need to poll.
while not resp:
  print("i0 polling")
  (resp, _) = mysvc_send.read()
print(f"i0 resp: {resp}")

recv = loopback.ports[esi.AppID("loopback_tohw")].write_port("recv")
recv.connect()
assert isinstance(recv.type, types.BitsType)

send = loopback.ports[esi.AppID("loopback_fromhw")].read_port("send")
send.connect()

data = 24
recv.write(int.to_bytes(data, 1, "little"))
resp = False
# Reads are non-blocking, so we need to poll.
resp_data: bytearray
while not resp:
  print("polling")
  (resp, resp_data) = send.read()
resp_int = int.from_bytes(resp_data, "little")

# Trace platform intentionally produces random responses.
if platform != "trace":
  print(f"data: {data}")
  print(f"resp: {resp_int}")
  assert resp_int == data

# Placeholder until we have a runtime function API.
myfunc = d.ports[esi.AppID("structFunc")]
myfunc.connect()

result = myfunc(a=10, b=-22)
print(f"result: {result}")
if platform != "trace":
  assert result == {"y": -22, "x": -21}

myfunc = d.ports[esi.AppID("arrayFunc")]
arg_chan = myfunc.write_port("arg").connect()
result_chan = myfunc.read_port("result").connect()

arg = [-22]
arg_chan.write(arg)

result: Optional[List[int]] = None
resp = False
while not resp:
  print("polling")
  (resp, result) = result_chan.read()

print(f"result: {result}")
if platform != "trace":
  assert result == [-21, -22]
print("PASS")
