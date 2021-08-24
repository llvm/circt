# RUN: %PYTHON% %s 2>&1 | FileCheck %s

from __future__ import annotations

import pycde
from pycde import (Input, Output, Parameter, module, externmodule, generator,
                   types, dim)
from circt.dialects import comb, hw
from circt.support import connect


@module
def PolynomialCompute(coefficients: Coefficients):

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
      return "PolyComputeForCoeff_" + '_'.join(
          [str(x) for x in coefficients.coeff])

    @generator
    def construct(mod):
      """Implement this module for input 'x'."""

      x = mod.x
      taps = list()
      for power, coeff in enumerate(coefficients.coeff):
        coeffVal = hw.ConstantOp.create(types.i32, coeff)
        if power == 0:
          newPartialSum = coeffVal.result
        else:
          partialSum = taps[-1]
          if power == 1:
            currPow = x
          else:
            x_power = [x for i in range(power)]
            currPow = comb.MulOp.create(*x_power)
          newPartialSum = comb.AddOp.create(
              partialSum, comb.MulOp.create(coeffVal, currPow))

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


class Coefficients:

  def __init__(self, coeff):
    self.coeff = coeff


@module
class PolynomialSystem:
  y = Output(types.i32)

  @generator
  def construct(_):
    i32 = types.i32
    x = hw.ConstantOp.create(i32, 23)
    poly = PolynomialCompute(Coefficients([62, 42, 6]))("example")
    connect(poly.x, x)
    PolynomialCompute(coefficients=Coefficients([62, 42, 6]))("example2",
                                                              x=poly.y)
    PolynomialCompute(Coefficients([1, 2, 3, 4, 5]))("example2", x=poly.y)

    cp = CoolPolynomialCompute([4, 42])
    cp.x.connect(23)
    return {"y": poly.y}


poly = pycde.System([PolynomialSystem])

print("Generating 1...")
poly.generate(iters=1)

print("Printing...")
poly.print()
# CHECK-LABEL:  hw.module @pycde.PolynomialSystem() -> (%y: i32)
# CHECK:    [[REG0:%.+]] = "pycde.PolynomialCompute"(%c23_i32) {instanceName = "example", opNames = ["x"], parameters = {coefficients = {coeff = [62, 42, 6]}, module_name = "PolyComputeForCoeff_62_42_6", unused_parameter = true}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    [[REG1:%.+]] = "pycde.PolynomialCompute"([[REG0]]) {instanceName = "example2", opNames = ["x"], parameters = {coefficients = {coeff = [62, 42, 6]}, module_name = "PolyComputeForCoeff_62_42_6", unused_parameter = true}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    [[REG2:%.+]] = "pycde.PolynomialCompute"([[REG0]]) {instanceName = "example2", opNames = ["x"], parameters = {coefficients = {coeff = [1, 2, 3, 4, 5]}, module_name = "PolyComputeForCoeff_1_2_3_4_5", unused_parameter = true}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    [[REG3:%.+]] = "pycde.CoolPolynomialCompute"(%c23_i32{{.*}}) {coefficients = [4, 42], opNames = ["x"], parameters = {}, resultNames = ["y"]} : (i32) -> i32
# CHECK:    hw.output [[REG0]] : i32

print("Generating 2...")
poly.generate(iters=1)

print("Printing...")
poly.print()
# CHECK-LABEL: hw.module @pycde.PolynomialSystem
# CHECK: %example.y = hw.instance "example" @PolyComputeForCoeff_62_42_6(%c23_i32) {parameters = {}} : (i32) -> i32
# CHECK: %example2.y = hw.instance "example2" @PolyComputeForCoeff_62_42_6(%example.y) {parameters = {}} : (i32) -> i32
# CHECK: %example2.y_0 = hw.instance "example2" @PolyComputeForCoeff_1_2_3_4_5(%example.y) {parameters = {}} : (i32) -> i32
# CHECK: %pycde.CoolPolynomialCompute.y = hw.instance "pycde.CoolPolynomialCompute" @supercooldevice(%c23_i32{{.*}}) {coefficients = [4, 42], parameters = {}} : (i32) -> i32
# CHECK-LABEL: hw.module @PolyComputeForCoeff_62_42_6(%x: i32) -> (%y: i32)
# CHECK: hw.constant 62
# CHECK: hw.constant 42
# CHECK: hw.constant 6
# CHECK-LABEL: hw.module @PolyComputeForCoeff_1_2_3_4_5(%x: i32) -> (%y: i32)
# CHECK: hw.constant 1
# CHECK: hw.constant 2
# CHECK: hw.constant 3
# CHECK: hw.constant 4
# CHECK: hw.constant 5
# CHECK-NOT: hw.module @pycde.PolynomialCompute

print("\n\n=== Verilog ===")
# CHECK-LABEL: === Verilog ===
poly.print_verilog()

# CHECK-LABEL:   module PolyComputeForCoeff_62_42_6(
# CHECK:    input  [31:0] x,
# CHECK:    output [31:0] y);
