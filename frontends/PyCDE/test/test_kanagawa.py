# RUN: %PYTHON% %s | FileCheck %s

from pycde import Module, System, generator
from pycde.common import Clock, Output, Reset
from pycde.types import Bits

from typing import Dict, Type

# Simple Kanagawa IR for basic test
basic_kanagawa_ir = """
kanagawa.design @test {
  kanagawa.container sym @basic_test {
    %out_port = kanagawa.port.output "out" sym @test_out : i32
    %c0_i32 = hw.constant 0 : i32
    kanagawa.port.write %out_port, %c0_i32 : !kanagawa.portref<out i32>
  }
}
"""


class TestKanagawaBasic(Module):
  """A simple Kanagawa module for testing."""

  clk = Clock()
  rst = Reset()
  out = Output(Bits(32))

  kanagawa_modules: Dict[str, Type[Module]]

  @generator
  def build(ports):
    kng_basic_mod = TestKanagawaBasic.kanagawa_modules["test_basic_test"]
    kng_basic_inst = kng_basic_mod()
    ports.out = kng_basic_inst.out


def test_kanagawa_basic():
  """Test basic Kanagawa IR import and lowering."""
  sys = System(TestKanagawaBasic)

  # Import the Kanagawa IR and run Kanagawa lowering passes from System class
  TestKanagawaBasic.kanagawa_modules = sys.import_mlir(
      basic_kanagawa_ir,
      name="kng_test",
      lowering=System.KANAGAWA_DIALECT_PASSES)
  sys.generate()
  sys.run_passes()
  print(sys.mod)


test_kanagawa_basic()

# CHECK-LABEL:  hw.module @TestKanagawaBasic(in %clk : i1, in %rst : i1, out out : i32) attributes {output_file = #hw.output_file<"TestKanagawaBasic.sv", includeReplicatedOps>} {
# CHECK-NEXT:     %test_basic_test.out = hw.instance "test_basic_test" sym @test_basic_test @test_basic_test() -> (out: i32)
# CHECK-NEXT:     hw.output %test_basic_test.out : i32
# CHECK-NEXT:   }
# CHECK-LABEL:  hw.module @test_basic_test(out out : i32) {
# CHECK-NEXT:     %c0_i32 = hw.constant 0 : i32
# CHECK-NEXT:     hw.output %c0_i32 : i32
# CHECK-NEXT:   }
