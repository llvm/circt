import esi

import random
import sys
from typing import List, Optional

platform = sys.argv[1]
acc_conn = esi.AcceleratorConnection(platform, sys.argv[2])
acc = acc_conn.build_accelerator()

add_bundle = acc.ports[esi.AppID("add")]
arg = add_bundle.write_port("arg")
arg.connect()
res = add_bundle.read_port("result")
res.connect()


def add_golden(a: int, b: int, arr: List[int]) -> int:
  return (a + b + (sum(arr) % 2**8) % 2**8)


for _ in range(10):
  data = {
      'a': random.randint(0, 2**4),
      'b': random.randint(0, 2**4),
      'arr': [random.randint(0, 2**4) for _ in range(16)]
  }
  expected = add_golden(**data)
  print(f"call({data})")
  arg.write(data)
  got_data = False
  resp: Optional[int] = None
  # Reads are non-blocking, so we need to poll.
  while not got_data:
    (got_data, resp) = res.read()
  if resp != expected:
    print(f"  = {resp} (expected {expected})")
  else:
    print(f"  = {resp} (matches Python result)")
