# REQUIRES: iverilog,cocotb
# RUN: %PYTHON% %s 2>&1 | FileCheck %s
from pycde import Input, Output, generator, modparams, Module, Clock
from pycde.types import Bit, Bits
from pycde.testing import cocotestbench, cocotest, cocoextra
from pycde.dialects import comb
import os

# CHECK:      ** TEST
# CHECK-NEXT: ********************************
# CHECK-NEXT: ** test_RegAdd.inc_test
# CHECK-NEXT: ** test_RegAdd.random_test
# CHECK-NEXT: ********************************
# CHECK-NEXT: ** TESTS=2 PASS=2 FAIL=0 SKIP=0
# CHECK-NEXT: ********************************


@modparams
def make_adder(width):

  class Adder(Module):
    in1 = Input(Bits(width))
    in2 = Input(Bits(width))
    out = Output(Bits(width))

    @generator
    def build(ports):
      ports.out = comb.AddOp(ports.in1, ports.in2)

  return Adder


class RegAdd(Module):
  rst = Input(Bit)
  clk = Clock()
  in1 = Input(Bits(16))
  in2 = Input(Bits(16))
  out = Output(Bits(16))

  @generator
  def build(ports):
    addRes = comb.AddOp(ports.in1, ports.in2)
    w16Adder = make_adder(16)(in1=ports.in1, in2=ports.in2)
    ports.out = w16Adder.out


@cocotestbench(RegAdd, simulator="icarus")
class RegAddTester:

  @cocotest
  async def random_test(ports):
    import cocotb
    import cocotb.clock
    from cocotb.triggers import FallingEdge
    import random

    # Create a 10us period clock on port clk
    clock = cocotb.clock.Clock(ports.clk, 10, units="us")
    cocotb.start_soon(clock.start())  # Start the clock

    for i in range(10):
      in1 = random.randint(0, 100)
      in2 = random.randint(0, 100)
      ports.in1.value = in1
      ports.in2.value = in2
      await FallingEdge(ports.clk)
      assert ports.out.value == (
          in1 + in2), "output q was incorrect on the {}th cycle".format(i)

  @cocotest
  async def inc_test(ports):
    import cocotb
    import cocotb.clock
    from cocotb.triggers import FallingEdge

    # Create a 10us period clock on port clk
    clock = cocotb.clock.Clock(ports.clk, 10, units="us")
    cocotb.start_soon(clock.start())  # Start the clock

    # Manual "reset", the FF reset isn't getting emitted in the sv...
    ports.in1.value = 0
    ports.in2.value = 0
    await FallingEdge(ports.clk)

    acc = 0
    for i in range(10):
      acc += 1
      ports.in1.value = ports.out.value
      ports.in2.value = 1
      await FallingEdge(ports.clk)
      assert ports.out.value == (
          acc), "output q was incorrect on the {}th cycle".format(i)


# -----

# CHECK:      ** TEST
# CHECK-NEXT: ********************************
# CHECK-NEXT: ** test_RegAdd.extern_test
# CHECK-NEXT: *******************************
# CHECK-NEXT: ** TESTS=1 PASS=1 FAIL=0 SKIP=0
# CHECK-NEXT: *******************************


class ExternAdder(Module):
  in1 = Input(Bits(16))
  in2 = Input(Bits(16))
  out = Output(Bits(16))

  module_name = "adder"


class RegAdd(Module):
  rst = Input(Bit)
  clk = Clock()
  in1 = Input(Bits(16))
  in2 = Input(Bits(16))
  out = Output(Bits(16))

  @generator
  def build(ports):
    w16Adder = ExternAdder(in1=ports.in1, in2=ports.in2)
    ports.out = w16Adder.out


@cocotestbench(RegAdd, simulator="icarus")
class RegAddTester:

  @cocoextra
  def extrafiles():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    return [os.path.join(scriptdir, "my_adder.sv")]

  @cocotest
  async def extern_test(ports):
    from cocotb.triggers import ReadOnly

    ports.in1.value = 1
    ports.in2.value = 2
    await ReadOnly()
    assert ports.out.value == (3), "Addition failed"
