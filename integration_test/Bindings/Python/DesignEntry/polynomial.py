# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt

from circt.design_entry import Input, Output, module, generator
from circt.esi import types
from circt.dialects import comb, hw

import sys


@module
class PolynomialCompute:
  """Module to compute ax^3 + bx^2 + cx + d for design-time coefficients"""

  # Evaluate polynomial for 'x'.
  x = Input(types.i32)

  def __init__(self, coefficients: list[int], **kwargs):
    """coefficients is in 'd' -> 'a' order."""
    self.__coefficients = coefficients
    # Full result.
    self.y = Output(types.i32)

  @generator
  def construct(mod):
    """Implement this module for input 'x'."""

    x = mod.x
    taps: list[mlir.ir.Value] = list()
    runningPower: list[mlir.ir.Value] = list()
    for power, coeff in enumerate([1, 2, 3]):
      coeffVal = hw.ConstantOp.create(types.i32, coeff)
      if power == 0:
        newPartialSum = coeffVal.result
      else:
        partialSum = taps[-1]
        if power == 1:
          currPow = x
        else:
          x_power = [x for i in range(power - 1)]
          currPow = comb.MulOp(types.i32, x_power + [runningPower[-1]]).result
        newPartialSum = comb.AddOp(
            types.i32,
            [
                partialSum,
                comb.MulOp(types.i32, [coeffVal.result, currPow]).result
            ]).result

        runningPower.append(currPow)

      taps.append(newPartialSum)

    # Final output
    return {"y": taps[-1]}


def build(top):
  i32 = mlir.ir.Type.parse("i32")
  x = hw.ConstantOp.create(i32, 23)
  poly = PolynomialCompute([62, 42, 6], x=x)
  hw.OutputOp([poly.y])


mod = mlir.ir.Module.create()
with mlir.ir.InsertionPoint(mod.body), circt.support.BackedgeBuilder():
  hw.HWModuleOp(name='top',
                input_ports=[],
                output_ports=[('y', mlir.ir.Type.parse("i32"))],
                body_builder=build)

mod.operation.print()
pm = mlir.passmanager.PassManager.parse("run-generators")
pm.run(mod)

mod.operation.print()
# CHECK:  hw.module @top() -> (%y: i32) {
# CHECK:    %c23_i32 = hw.constant 23 : i32
# CHECK:    [[REG0:%.+]] = "circt.PolynomialCompute"(%c23_i32) {opNames = ["x"], resultNames = ["y"]} : (i32) -> i32
# CHECK:    hw.output [[REG0]] : i32

print("\n\n=== Verilog ===")
# CHECK-LABEL: === Verilog ===

pm = mlir.passmanager.PassManager.parse(
  "hw-legalize-names,hw.module(hw-cleanup)")
pm.run(mod)
circt.export_verilog(mod, sys.stdout)
# CHECK:  module circt_PolynomialCompute(
# CHECK:    input  [31:0] x,
# CHECK:    output [31:0] y);
# CHECK:    assign y = 32'h1 + 32'h2 * x + 32'h3 * x * x;
