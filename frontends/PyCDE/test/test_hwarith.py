# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from math import fabs
from yaml import emit
from pycde import Input, Output, generator
from pycde.testing import unittestmodule
from pycde.dialects import comb
from pycde.pycde_types import dim, types


# CHECK: msft.module @InfixArith {} (%in0: si16, %in1: ui16)
# CHECK-NEXT:   %0 = hwarith.add %in0, %in1 : (si16, ui16) -> si18
# CHECK-NEXT:   %1 = hwarith.sub %in0, %in1 : (si16, ui16) -> si18
# CHECK-NEXT:   %2 = hwarith.mul %in0, %in1 : (si16, ui16) -> si32
# CHECK-NEXT:   %3 = hwarith.div %in0, %in1 : (si16, ui16) -> si16
# CHECK-NEXT:   %c-1_i16 = hw.constant -1 : i16
# CHECK-NEXT:   %4 = hwarith.cast %c-1_i16 : (i16) -> si16
# CHECK-NEXT:   %5 = hwarith.mul %in0, %4 : (si16, si16) -> si32
# CHECK-NEXT:   msft.output
@unittestmodule(run_passes=False)
class InfixArith:
  in0 = Input(types.si16)
  in1 = Input(types.ui16)

  @generator
  def construct(ports):
    add = ports.in0 + ports.in1
    sub = ports.in0 - ports.in1
    mul = ports.in0 * ports.in1
    div = ports.in0 / ports.in1
    neg = -ports.in0


# -----


# CHECK: msft.module @InfixLogic {} (%in0: si16, %in1: ui16)
# CHECK-NEXT:  %0 = hwarith.cast %in0 : (si16) -> i16
# CHECK-NEXT:  %1 = hwarith.cast %in1 : (ui16) -> i16
# CHECK-NEXT:  %2 = comb.and %0, %1 : i16
# CHECK-NEXT:  %3 = hwarith.cast %in0 : (si16) -> i16
# CHECK-NEXT:  %4 = hwarith.cast %in1 : (ui16) -> i16
# CHECK-NEXT:  %5 = comb.or %3, %4 : i16
# CHECK-NEXT:  %6 = hwarith.cast %in0 : (si16) -> i16
# CHECK-NEXT:  %7 = hwarith.cast %in1 : (ui16) -> i16
# CHECK-NEXT:  %8 = comb.xor %6, %7 : i16
# CHECK-NEXT:  %9 = hwarith.cast %in0 : (si16) -> i16
# CHECK-NEXT:  %c-1_i16 = hw.constant -1 : i16
# CHECK-NEXT:  %10 = comb.xor %9, %c-1_i16 : i16
# CHECK-NEXT:  msft.output
@unittestmodule(run_passes=False)
class InfixLogic:
  in0 = Input(types.si16)
  in1 = Input(types.ui16)

  @generator
  def construct(ports):
    and_ = ports.in0 & ports.in1
    or_ = ports.in0 | ports.in1
    xor = ports.in0 ^ ports.in1
    inv = ~ports.in0


# -----


# CHECK: msft.module @InfixComparison {} (%in0: i16, %in1: i16)
# CHECK-NEXT:    %0 = comb.icmp eq %in0, %in1 : i16
# CHECK-NEXT:    %1 = comb.icmp ne %in0, %in1 : i16
# CHECK-NEXT:    msft.output
@unittestmodule(run_passes=False)
class InfixComparison:
  in0 = Input(types.i16)
  in1 = Input(types.i16)

  @generator
  def construct(ports):
    eq = ports.in0 == ports.in1
    neq = ports.in0 != ports.in1


# -----


# CHECK:  msft.module @Multiple {} (%in0: si16, %in1: si16) -> (out0: i16)
# CHECK-NEXT:    %0 = hwarith.add %in0, %in1 : (si16, si16) -> si17
# CHECK-NEXT:    %1 = hwarith.add %0, %in0 : (si17, si16) -> si18
# CHECK-NEXT:    %2 = hwarith.add %1, %in1 : (si18, si16) -> si19
# CHECK-NEXT:    %3 = hwarith.cast %2 : (si19) -> i16
# CHECK-NEXT:    msft.output %3 : i16
@unittestmodule(run_passes=False)
class Multiple:
  in0 = Input(types.si16)
  in1 = Input(types.si16)
  out0 = Output(types.i16)

  @generator
  def construct(ports):
    ports.out0 = (ports.in0 + ports.in1 + ports.in0 + ports.in1).as_int(16)


# -----


# CHECK:  msft.module @Casting {} (%in0: i16)
# CHECK-NEXT:    %0 = hwarith.cast %in0 : (i16) -> si16
# CHECK-NEXT:    %1 = hwarith.cast %in0 : (i16) -> ui16
# CHECK-NEXT:    %2 = hwarith.cast %0 : (si16) -> i16
# CHECK-NEXT:    %3 = hwarith.cast %in0 : (i16) -> si8
# CHECK-NEXT:    %4 = hwarith.cast %in0 : (i16) -> ui8
# CHECK-NEXT:    %5 = hwarith.cast %0 : (si16) -> i8
# CHECK-NEXT:    %6 = hwarith.cast %0 : (si16) -> si24
# CHECK-NEXT:    msft.output
@unittestmodule(run_passes=False)
class Casting:
  in0 = Input(types.i16)

  @generator
  def construct(ports):
    in0s = ports.in0.as_sint()
    in0u = ports.in0.as_uint()
    in0s_i = in0s.as_int()
    in0s8 = ports.in0.as_sint(8)
    in0u8 = ports.in0.as_uint(8)
    in0s_i8 = in0s.as_int(8)
    in0s_s24 = in0s.as_sint(24)


# -----


# CHECK: hw.module @Lowering<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%in0: i16, %in1: i16) -> (out0: i16)
# CHECK-NEXT:    %0 = comb.extract %in0 from 15 : (i16) -> i1
# CHECK-NEXT:    %1 = comb.concat %0, %in0 : i1, i16
# CHECK-NEXT:    %2 = comb.extract %in1 from 15 : (i16) -> i1
# CHECK-NEXT:    %3 = comb.concat %2, %in1 : i1, i16
# CHECK-NEXT:    %4 = comb.add %1, %3 : i17
# CHECK-NEXT:    %5 = comb.extract %4 from 0 : (i17) -> i16
# CHECK-NEXT:    hw.output %5 : i16
@unittestmodule(generate=True, run_passes=True, print_after_passes=True)
class Lowering:
  in0 = Input(types.i16)
  in1 = Input(types.i16)
  out0 = Output(types.i16)

  @generator
  def construct(ports):
    ports.out0 = (ports.in0.as_sint() + ports.in1.as_sint()).as_int(16)
