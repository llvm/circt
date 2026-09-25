# REQUIRES: iverilog-12,cocotb
# RUN: %PYTHON% %s 2>&1 | FileCheck %s

# Struct type aliases are declared in a SystemVerilog package. Check that the
# package is compiled and its structs can be used in module ports.

from pycde import Input, Output, generator, Module
from pycde.signals import Struct
from pycde.types import UInt
from pycde.testing import cocotestbench, cocotest


class Pair(Struct):
  a: UInt(8)
  b: UInt(8)


class PairSum(Module):
  inp = Input(Pair)
  passthrough = Output(Pair)
  sum = Output(UInt(9))

  @generator
  def build(ports):
    ports.passthrough = ports.inp
    ports.sum = ports.inp.a + ports.inp.b


# CHECK:      ** TEST
# CHECK:      ** test_PairSum.sum_test
# CHECK:      ** TESTS=1 PASS=1 FAIL=0 SKIP=0


@cocotestbench(PairSum, simulator="icarus")
class PairSumTester:

  @cocotest
  async def sum_test(ports):
    from cocotb.triggers import Timer

    # `a` is the most significant field of the packed struct.
    ports.inp.value = (0x12 << 8) | 0x34
    await Timer(1, "ns")
    assert ports.passthrough.value == (0x12 << 8) | 0x34
    assert ports.sum.value == 0x12 + 0x34
