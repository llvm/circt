# RUN: %PYTHON% %s | FileCheck %s

from __future__ import annotations

import mlir
import circt

import pycde
from pycde import Input, Output, Parameter, module, generator
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
    self.coefficients = Parameter(coefficients)
    # Full result.
    self.y = Output(types.i32)

  @generator
  def construct(mod, params):
    """Implement this module for input 'x'."""

    x = mod.x
    taps: list[mlir.ir.Value] = list()
    # TODO: use the coefficient parameter, once its usable.
    for power, coeff in enumerate([62, 42, 6]):
      coeffVal = hw.ConstantOp.create(types.i32, coeff)
      if power == 0:
        newPartialSum = coeffVal.result
      else:
        partialSum = taps[-1]
        if power == 1:
          currPow = x
        else:
          x_power = [x for i in range(power)]
          currPow = comb.MulOp(types.i32, x_power).result
        newPartialSum = comb.AddOp(types.i32, [
            partialSum,
            comb.MulOp(types.i32, [coeffVal.result, currPow]).result
        ]).result

      taps.append(newPartialSum)

    # Final output
    return {"y": taps[-1]}


class Polynomial(pycde.System):
  inputs = []
  outputs = [('y', types.i32)]

  def build(self, top):
    i32 = types.i32
    x = hw.ConstantOp.create(i32, 23)
    poly = PolynomialCompute([62, 42, 6], x=x)
    hw.OutputOp([poly.y])


poly = Polynomial()

poly.print()
# CHECK:  hw.module @top() -> (%y: i32) {
# CHECK:    %c23_i32 = hw.constant 23 : i32
# CHECK:    [[REG0:%.+]] = "pycde.PolynomialCompute"(%c23_i32) {coefficients = [62, 42, 6], opNames = ["x"], resultNames = ["y"]} : (i32) -> i32
# CHECK:    hw.output [[REG0]] : i32

poly.generate()
poly.print()

print("\n\n=== Verilog ===")
# CHECK-LABEL: === Verilog ===
poly.print_verilog()

# CHECK:  module pycde_PolynomialCompute(
# CHECK:    input  [31:0] x,
# CHECK:    output [31:0] y);
# CHECK:    assign y = 32'h3E + 32'h2A * x + 32'h6 * x * x;
