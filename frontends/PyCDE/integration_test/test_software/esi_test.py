from typing import Optional
import esiaccel as esi
from esiaccel.types import MMIORegion

import sys
import time

ctxt = esi.Context(esi.LogLevel.Debug)
platform = sys.argv[1]
conn_str = sys.argv[2]
acc = ctxt.connect(platform, conn_str)

mmio = acc.get_service_mmio()
data = mmio.read(8)
assert data == 0x207D98E5E5100E51

assert acc.sysinfo().esi_version() == 0
m = acc.manifest()
assert m.api_version == 0
print(m.type_table)

# Test the cycle count and clock frequency APIs
sysinfo = acc.sysinfo()
cycle_count = sysinfo.cycle_count()
if cycle_count is not None:
  print(f"Cycle count: {cycle_count}")
  assert cycle_count > 0, f"Cycle count should be positive, got {cycle_count}"

  # Test that cycle count is monotonically increasing
  time.sleep(0.01)  # Small delay to let simulation advance
  cycle_count2 = sysinfo.cycle_count()
  print(f"Cycle count after delay: {cycle_count2}")
  assert cycle_count2 > cycle_count, \
      f"Cycle count should be monotonically increasing: {cycle_count2} <= {cycle_count}"

  # Test again to ensure consistency
  time.sleep(0.01)
  cycle_count3 = sysinfo.cycle_count()
  print(f"Cycle count after second delay: {cycle_count3}")
  assert cycle_count3 > cycle_count2, \
      f"Cycle count should be monotonically increasing: {cycle_count3} <= {cycle_count2}"
else:
  print("Cycle count: not available")

clock_freq = sysinfo.core_clock_frequency()
print(f"Clock frequency: {clock_freq} Hz")
if platform == "cosim":
  assert clock_freq == 20_000_000, \
      f"Expected clock frequency 20_000_000 Hz for cosim, got {clock_freq}"
else:
  if clock_freq is not None:
    print(f"Core clock frequency: {clock_freq} Hz")
    assert clock_freq > 0, f"Clock frequency should be positive, got {clock_freq}"
  else:
    print("Core clock frequency: not available")

d = acc.build_accelerator()

mmio_svc: esi.accelerator.MMIO
for svc in d.services:
  if isinstance(svc, esi.accelerator.MMIO):
    mmio_svc = svc
    break

for id, region in mmio_svc.regions.items():
  print(f"Region {id}: {region.base} - {region.base + region.size}")

assert len(mmio_svc.regions) == 5

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
# Callback tests
################################################################################

callback = d.children[esi.AppID("callback")]
cb_port = callback.ports[esi.AppID("cb")]
cb_mmio = callback.ports[esi.AppID("cmd")]

recv_data: Optional[int] = None


def my_callback(data: int) -> int:
  global recv_data
  recv_data = data
  print(f"Callback received data: {data}")
  return data + 7


cb_port.connect(my_callback)
cb_mmio.write(0x10, 5)
while recv_data is None:
  time.sleep(0.25)
assert recv_data == 5

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

################################################################################
# StructToWindowFunc tests
################################################################################

print("Testing StructToWindowFunc...")
struct_to_window_bundle = d.ports[esi.AppID("struct_to_window")]

# Get the write port for sending the complete struct
struct_send = struct_to_window_bundle.write_port("arg")
struct_send.connect()

# Get the read port for receiving the windowed result
window_recv = struct_to_window_bundle.read_port("result")
window_recv.connect()

# Create test data - a struct with four 32-bit fields (as bytearrays, little-endian)
test_struct = {
    "a": bytearray([0x11, 0x11, 0x11, 0x11]),
    "b": bytearray([0x22, 0x22, 0x22, 0x22]),
    "c": bytearray([0x33, 0x33, 0x33, 0x33]),
    "d": bytearray([0x44, 0x44, 0x44, 0x44])
}

# Send the complete struct
struct_send.write(test_struct)

# The windowed result should arrive as two frames
# Frame 1 contains fields a and b
# Frame 2 contains fields c and d
# After translation, we should get back the complete struct
result = window_recv.read()

print(f"Sent struct: {test_struct}")
print(f"Received result: {result}")

# Verify the result matches the input
assert result["a"] == test_struct[
    "a"], f"Field 'a' mismatch: {result['a']} != {test_struct['a']}"
assert result["b"] == test_struct[
    "b"], f"Field 'b' mismatch: {result['b']} != {test_struct['b']}"
assert result["c"] == test_struct[
    "c"], f"Field 'c' mismatch: {result['c']} != {test_struct['c']}"
assert result["d"] == test_struct[
    "d"], f"Field 'd' mismatch: {result['d']} != {test_struct['d']}"

print("PASS: StructToWindowFunc test passed")

# Test with different values
test_struct2 = {
    "a": bytearray([0xEF, 0xBE, 0xAD, 0xDE]),  # 0xDEADBEEF little-endian
    "b": bytearray([0xBE, 0xBA, 0xFE, 0xCA]),  # 0xCAFEBABE little-endian
    "c": bytearray([0x78, 0x56, 0x34, 0x12]),  # 0x12345678 little-endian
    "d":
        bytearray([0x21, 0x43, 0x65, 0x87])  # 0x87654321 little-endian
}
struct_send.write(test_struct2)
result2 = window_recv.read()

print(f"Sent struct: {test_struct2}")
print(f"Received result: {result2}")

assert result2["a"] == test_struct2[
    "a"], f"Field 'a' mismatch: {result2['a']} != {test_struct2['a']}"
assert result2["b"] == test_struct2[
    "b"], f"Field 'b' mismatch: {result2['b']} != {test_struct2['b']}"
assert result2["c"] == test_struct2[
    "c"], f"Field 'c' mismatch: {result2['c']} != {test_struct2['c']}"
assert result2["d"] == test_struct2[
    "d"], f"Field 'd' mismatch: {result2['d']} != {test_struct2['d']}"

print("PASS: StructToWindowFunc test 2 passed")

struct_send.disconnect()
window_recv.disconnect()

################################################################################
# WindowToStructFunc tests
################################################################################

print("Testing WindowToStructFunc...")
window_to_struct_bundle = d.ports[esi.AppID("struct_from_window")]

# Get the write port for sending the windowed struct (two frames)
window_send = window_to_struct_bundle.write_port("arg")
window_send.connect()

# Get the read port for receiving the complete struct
struct_recv = window_to_struct_bundle.read_port("result")
struct_recv.connect()

# Create test data - a struct with four 32-bit fields (as bytearrays, little-endian)
# We'll send this as a windowed struct and expect to get it back as a complete struct
test_window_struct = {
    "a": bytearray([0xAA, 0xAA, 0xAA, 0xAA]),
    "b": bytearray([0xBB, 0xBB, 0xBB, 0xBB]),
    "c": bytearray([0xCC, 0xCC, 0xCC, 0xCC]),
    "d": bytearray([0xDD, 0xDD, 0xDD, 0xDD])
}

# Send the windowed struct (the runtime will split it into two frames)
window_send.write(test_window_struct)

# Read the complete struct result
result = struct_recv.read()

print(f"Sent windowed struct: {test_window_struct}")
print(f"Received complete struct: {result}")

# Verify the result matches the input
assert result["a"] == test_window_struct[
    "a"], f"Field 'a' mismatch: {result['a']} != {test_window_struct['a']}"
assert result["b"] == test_window_struct[
    "b"], f"Field 'b' mismatch: {result['b']} != {test_window_struct['b']}"
assert result["c"] == test_window_struct[
    "c"], f"Field 'c' mismatch: {result['c']} != {test_window_struct['c']}"
assert result["d"] == test_window_struct[
    "d"], f"Field 'd' mismatch: {result['d']} != {test_window_struct['d']}"

print("PASS: WindowToStructFunc test passed")

# Test with different values
test_window_struct2 = {
    "a": bytearray([0x01, 0x02, 0x03, 0x04]),
    "b": bytearray([0x05, 0x06, 0x07, 0x08]),
    "c": bytearray([0x09, 0x0A, 0x0B, 0x0C]),
    "d": bytearray([0x0D, 0x0E, 0x0F, 0x10])
}
window_send.write(test_window_struct2)
result2 = struct_recv.read()

print(f"Sent windowed struct: {test_window_struct2}")
print(f"Received complete struct: {result2}")

assert result2["a"] == test_window_struct2[
    "a"], f"Field 'a' mismatch: {result2['a']} != {test_window_struct2['a']}"
assert result2["b"] == test_window_struct2[
    "b"], f"Field 'b' mismatch: {result2['b']} != {test_window_struct2['b']}"
assert result2["c"] == test_window_struct2[
    "c"], f"Field 'c' mismatch: {result2['c']} != {test_window_struct2['c']}"
assert result2["d"] == test_window_struct2[
    "d"], f"Field 'd' mismatch: {result2['d']} != {test_window_struct2['d']}"

print("PASS: WindowToStructFunc test 2 passed")

window_send.disconnect()
struct_recv.disconnect()
