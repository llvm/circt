# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

from circt import register_dialects
from circt.dialects import firrtl

from circt.ir import Context, InsertionPoint, IntegerAttr, IntegerType, Location, Module

from functools import reduce
from typing import Sequence


def fir_filter(bit_width: int, coeffs: Sequence[int]):
  out_width = bit_width * 2 + max(1, len(coeffs) - 1)
  with firrtl.CircuitOp("fir_filter"):
    with firrtl.FModuleOp("fir_filter",
                          clock=firrtl.Input(firrtl.Clock()),
                          inp=firrtl.Input(firrtl.UInt(bit_width)),
                          out=firrtl.Output(firrtl.UInt(out_width))) as module:
      # Create the serial-in, parallel-out shift register
      zs = firrtl.RegOp("zs", module.clock,
                        firrtl.Vec(len(coeffs), firrtl.UInt(bit_width)))

      z0 = firrtl.SubindexOp(zs, 0)
      firrtl.StrictConnectOp(z0, module.inp)

      for i in range(1, len(coeffs)):
        z_next = firrtl.SubindexOp(zs, i)
        z_last = firrtl.SubindexOp(zs, i - 1)
        firrtl.StrictConnectOp(z_next, z_last)

      # Do the multiplies
      products = [
          firrtl.MulPrimOp(firrtl.SubindexOp(zs, i),
                           firrtl.ConstantOp(coeffs[i], firrtl.UInt(bit_width)))
          for i in range(len(coeffs))
      ]

      # Sum up the products
      sum = reduce(lambda a, b: firrtl.AddPrimOp(a, b), products)

      firrtl.StrictConnectOp(module.out, sum)


with Context() as ctx, Location.unknown():
  register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    fir_filter(8, [1, 1, 1])
    fir_filter(8, [0, 1])
    fir_filter(8, [1, 2, 3, 2, 1])

  m.operation.verify()

  print(m)
