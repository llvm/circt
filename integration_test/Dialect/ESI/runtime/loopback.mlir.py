from typing import List, Optional
import esiaccel
import esiaccel.types as types
import sys
import time

platform = sys.argv[1]
acc = esiaccel.AcceleratorConnection(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 0
m = acc.manifest()
assert m.api_version == 0

for esiType in m.type_table:
  print(f"{esiType}")

d = acc.build_accelerator()

loopback = d.children[esiaccel.AppID("loopback_inst", 0)]
appid = loopback.id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0

mysvc_send = loopback.ports[esiaccel.AppID("mysvc_recv")].write_port("recv")
mysvc_send.connect()
mysvc_send.write(None)
print(f"mysvc_send.type: {mysvc_send.type}")
assert isinstance(mysvc_send.type, types.VoidType)

mysvc_recv = loopback.ports[esiaccel.AppID("mysvc_send")].read_port("send")
mysvc_recv.connect()
mysvc_recv.read()
print("mysvc_recv.read() returned")

recv = loopback.ports[esiaccel.AppID("loopback_tohw")].write_port("recv")
recv.connect()
assert isinstance(recv.type, types.BitsType)

send = loopback.ports[esiaccel.AppID("loopback_fromhw")].read_port("send")
send.connect()
assert isinstance(send.type, types.BitsType)

data = 24
recv.write(int.to_bytes(data, 1, "little"))
resp_data: bytearray = send.read()
resp_int = int.from_bytes(resp_data, "little")

print(f"data: {data}")
print(f"resp: {resp_int}")
# Trace platform intentionally produces random responses.
if platform != "trace":
  assert resp_int == data

# Placeholder until we have a runtime function API.
myfunc = d.ports[esiaccel.AppID("structFunc")]
myfunc.connect()

for _ in range(10):
  future_result = myfunc(a=10, b=-22)
  result = future_result.result()

  print(f"result: {result}")
  if platform != "trace":
    assert result == {"y": -22, "x": -21}

if platform != "trace":
  print("Checking function call result ordering.")
  future_result1 = myfunc(a=15, b=-32)
  future_result2 = myfunc(a=32, b=47)
  result2 = future_result2.result()
  result1 = future_result1.result()
  print(f"result1: {result1}")
  print(f"result2: {result2}")
  assert result1 == {"y": -32, "x": -31}, "result1 is incorrect"
  assert result2 == {"y": 47, "x": 48}, "result2 is incorrect"

myfunc = d.ports[esiaccel.AppID("arrayFunc")]
arg_chan = myfunc.write_port("arg").connect()
result_chan = myfunc.read_port("result").connect()

arg = [-22]
arg_chan.write(arg)

result: List[int] = result_chan.read()

print(f"result: {result}")
if platform != "trace":
  assert result == [-21, -22]
print("PASS")
