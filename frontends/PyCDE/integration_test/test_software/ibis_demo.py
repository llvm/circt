import esi

import crc
import random
import sys
from typing import List

platform = sys.argv[1]
acc_conn = esi.AcceleratorConnection(platform, sys.argv[2])
acc = acc_conn.build_accelerator()

print("***** Testing add function")

add = acc.ports[esi.AppID("add")]
add.connect()


def add_golden(a: int, b: int, arr: List[int]) -> int:
  return (a + b + sum(arr)) % 2**8


for _ in range(3):
  a = random.randint(0, 2**8 - 1)
  b = random.randint(0, 2**8 - 1)
  arr = [random.randint(0, 2**8 - 1) for _ in range(16)]

  expected = add_golden(a=a, b=b, arr=arr)
  print(f"call(a={a}, b={b}, arr={arr})")

  resp = add(a=a, b=b, arr=arr)
  if resp != expected:
    print(f"  = {resp} (expected {expected})")
  else:
    print(f"  = {resp} (matches Python result)")

print()
print("***** Testing compute_crc function")

golden_crc = crc.Calculator(crc.Crc32.POSIX)

compute_crc = acc.ports[esi.AppID("crc")]
compute_crc.connect()

data = [random.randint(0, 2**4) for _ in range(64)]
crc = compute_crc(identifier=0, input=data, input_bytes=64, reset=1)
print(f"crc({data})")
print(f"  = 0x{crc:x}")
print(f"    0x{golden_crc.checksum(data):x} (expected)")

new_data = [random.randint(0, 2**4) for _ in range(64)]
data.append(new_data)
crc = compute_crc(identifier=0, input=new_data, input_bytes=64, reset=0)
print(f"crc({new_data})")
print(f"  = 0x{crc:x}")
print(f"    0x{golden_crc.checksum(data):x} (expected)")
