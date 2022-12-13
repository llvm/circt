# RUN: %PYTHON% %s | FileCheck %s

from pycde.dialects import comb, hw
from pycde import dim, generator, Input, Output
from pycde.testing import unittestmodule


@unittestmodule(SIZE=4)
def MyModule(SIZE: int):

  class Mod:
    inp = Input(dim(SIZE))
    out = Output(dim(SIZE))

    @generator
    def construct(mod):
      c1 = hw.ConstantOp(dim(SIZE), 1)
      # CHECK: %[[EQ:.+]] = comb.icmp bin eq
      eq = comb.EqOp(c1, mod.inp)
      # CHECK: %[[A1:.+]] = hw.array_create %[[EQ]], %[[EQ]]
      a1 = hw.ArrayCreateOp([eq, eq])
      # CHECK: %[[A2:.+]] = hw.array_create %[[EQ]], %[[EQ]]
      a2 = hw.ArrayCreateOp([eq, eq])
      # CHECK: %[[COMBINED:.+]] = hw.array_concat %[[A1]], %[[A2]]
      combined = hw.ArrayConcatOp(a1, a2)
      mod.out = hw.BitcastOp(dim(SIZE), combined)

  return Mod
