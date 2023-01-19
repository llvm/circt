# RUN: %PYTHON% %s | FileCheck %s

from pycde.dialects import comb, hw
from pycde import dim, generator, types, Input, Output, Module
from pycde.value import And, Or
from pycde.testing import unittestmodule


@unittestmodule(SIZE=4)
def MyModule(SIZE: int):

  class Mod(Module):
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


# CHECK-LABEL: msft.module @Mod {} (%inp: !hw.array<5xi1>)
@unittestmodule()
class Mod(Module):
  inp = Input(dim(types.i1, 5))

  @generator
  def construct(ports):
    # CHECK:  %c0_i3 = hw.constant 0 : i3
    # CHECK:  %0 = hw.array_get %inp[%c0_i3] {sv.namehint = "inp__0"} : !hw.array<5xi1>, i3
    # CHECK:  %c1_i3 = hw.constant 1 : i3
    # CHECK:  %1 = hw.array_get %inp[%c1_i3] {sv.namehint = "inp__1"} : !hw.array<5xi1>, i3
    # CHECK:  %c2_i3 = hw.constant 2 : i3
    # CHECK:  %2 = hw.array_get %inp[%c2_i3] {sv.namehint = "inp__2"} : !hw.array<5xi1>, i3
    # CHECK:  %c3_i3 = hw.constant 3 : i3
    # CHECK:  %3 = hw.array_get %inp[%c3_i3] {sv.namehint = "inp__3"} : !hw.array<5xi1>, i3
    # CHECK:  %c-4_i3 = hw.constant -4 : i3
    # CHECK:  %4 = hw.array_get %inp[%c-4_i3] {sv.namehint = "inp__4"} : !hw.array<5xi1>, i3
    # CHECK:  %5 = comb.or bin %0, %1, %2, %3, %4 : i1
    ports.inp.or_reduce()

    # CHECK:  %c0_i3_0 = hw.constant 0 : i3
    # CHECK:  %6 = hw.array_get %inp[%c0_i3_0] {sv.namehint = "inp__0"} : !hw.array<5xi1>, i3
    # CHECK:  %c1_i3_1 = hw.constant 1 : i3
    # CHECK:  %7 = hw.array_get %inp[%c1_i3_1] {sv.namehint = "inp__1"} : !hw.array<5xi1>, i3
    # CHECK:  %c2_i3_2 = hw.constant 2 : i3
    # CHECK:  %8 = hw.array_get %inp[%c2_i3_2] {sv.namehint = "inp__2"} : !hw.array<5xi1>, i3
    a, b, c = ports.inp[0], ports.inp[1], ports.inp[2]

    # CHECK:  %9 = comb.or bin %6, %7, %8 : i1
    Or(a, b, c)

    # CHECK:  %10 = comb.and bin %6, %7, %8 : i1
    And(a, b, c)
