# RUN: %PYTHON% %s 2>&1 | FileCheck %s

from __future__ import annotations

import mlir

import pycde
from pycde import (Input, Output, Parameter, module, externmodule, generator,
                   types)
from circt.dialects import comb, hw


@module
def PolynomialCompute(coefficients):

  class PolynomialCompute:
    """Module to compute ax^3 + bx^2 + cx + d for design-time coefficients"""

    # Evaluate polynomial for 'x'.
    x = Input(types.i32)
    y = Output(types.int(8 * 4))

    unused_parameter = Parameter(True)

    def __init__(self, name: str):
      """coefficients is in 'd' -> 'a' order."""
      self.instanceName = name

    @staticmethod
    def get_module_name():
      return "PolyComputeForCoeff_" + '_'.join([str(x) for x in coefficients])

    @generator
    def construct(mod):
      """Implement this module for input 'x'."""

      x = mod.x
      taps: list[mlir.ir.Value] = list()
      for power, coeff in enumerate(coefficients):
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

  return PolynomialCompute


@externmodule("supercooldevice")
class CoolPolynomialCompute:
  x = Input(types.i32)
  y = Output(types.i32)

  def __init__(self, coefficients):
    self.coefficients = coefficients


class Polynomial(pycde.System):
  inputs = []
  outputs = [('y', types.i32)]

  def build(self, top):
    i32 = types.i32
    x = hw.ConstantOp.create(i32, 23)
    poly = PolynomialCompute([62, 42, 6])("example", x=x)
    PolynomialCompute(coefficients=[62, 42, 6])("example2", x=poly.y)
    PolynomialCompute([1, 2, 3, 4, 5])("example2", x=poly.y)

    CoolPolynomialCompute([4, 42], x=x)
    hw.OutputOp([poly.y])


poly = Polynomial()

poly.graph()
# CHECK-LABEL: digraph "top"
# CHECK: label="top";
# CHECK: [shape=record,label="{hw.constant\ni32\n\nvalue: 23 : i32}"];

poly.print()
# CHECK-LABEL:  hw.module @top() -> (%y: i32)
# CHECK:    [[REG0:%.+]] = "pycde.PolynomialCompute"(%c23_i32) {instanceName = "example", opNames = ["x"], parameters = {coefficients = [62, 42, 6], module_name = "PolyComputeForCoeff_62_42_6", unused_parameter = true}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    [[REG1:%.+]] = "pycde.PolynomialCompute"([[REG0]]) {instanceName = "example2", opNames = ["x"], parameters = {coefficients = [62, 42, 6], module_name = "PolyComputeForCoeff_62_42_6", unused_parameter = true}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    [[REG2:%.+]] = "pycde.PolynomialCompute"([[REG0]]) {instanceName = "example2", opNames = ["x"], parameters = {coefficients = [1, 2, 3, 4, 5], module_name = "PolyComputeForCoeff_1_2_3_4_5", unused_parameter = true}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    [[REG3:%.+]] = "pycde.CoolPolynomialCompute"(%c23_i32) {coefficients = [4, 42], opNames = ["x"], parameters = {}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    hw.output [[REG0]] : i32

poly.generate()
poly.print()
# CHECK-LABEL: hw.module @top
# CHECK: %example.y = hw.instance "example" @PolyComputeForCoeff_62_42_6(%c23_i32) {parameters = {}} : (i32) -> i32
# CHECK: %example2.y = hw.instance "example2" @PolyComputeForCoeff_62_42_6(%example.y) {parameters = {}} : (i32) -> i32
# CHECK: %example2.y_0 = hw.instance "example2" @PolyComputeForCoeff_1_2_3_4_5(%example.y) {parameters = {}} : (i32) -> i32
# CHECK: %pycde.CoolPolynomialCompute.y = hw.instance "pycde.CoolPolynomialCompute" @supercooldevice(%c23_i32) {coefficients = [4, 42], parameters = {}} : (i32) -> i32
# CHECK: hw.module @PolyComputeForCoeff_62_42_6(%x: i32) -> (%y: i32)
# CHECK: hw.module @PolyComputeForCoeff_1_2_3_4_5(%x: i32) -> (%y: i32)
# CHECK-NOT: hw.module @pycde.PolynomialCompute

print("\n\n=== Verilog ===")
# CHECK-LABEL: === Verilog ===
poly.print_verilog()

# CHECK-LABEL:   module PolyComputeForCoeff_62_42_6(
# CHECK:    input  [31:0] x,
# CHECK:    output [31:0] y);
