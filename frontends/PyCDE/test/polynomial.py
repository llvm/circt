# RUN: %PYTHON% %s 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file PolynomialSystem/PolynomialSystem.sv --check-prefix=OUTPUT

from __future__ import annotations

import pycde
from pycde import (Input, Output, module, externmodule, generator, types, dim)
from circt.dialects import comb, hw
from circt.support import connect


@module
def PolynomialCompute(coefficients: Coefficients):

  class PolynomialCompute:
    """Module to compute ax^3 + bx^2 + cx + d for design-time coefficients"""

    # Evaluate polynomial for 'x'.
    x = Input(types.i32)
    y = Output(types.int(8 * 4))

    def __init__(self, name: str):
      """coefficients is in 'd' -> 'a' order."""
      self.instance_name = name

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
      mod.y = taps[-1]

  return PolynomialCompute


@externmodule("supercooldevice")
class CoolPolynomialCompute:
  x = Input(types.i32)
  y = Output(types.i32)

  def __init__(self, coefficients):
    self.coefficients = coefficients


@externmodule("parameterized_extern")
def ExternWithParams(a, b):

  class M:
    pass

  return M


class Coefficients:

  def __init__(self, coeff):
    self.coeff = coeff


@module
class PolynomialSystem:
  y = Output(types.i32)

  @generator
  def construct(ports):
    i32 = types.i32
    x = hw.ConstantOp.create(i32, 23)
    poly = PolynomialCompute(Coefficients([62, 42, 6]))("example")
    connect(poly.x, x)
    PolynomialCompute(coefficients=Coefficients([62, 42, 6]))("example2",
                                                              x=poly.y)
    PolynomialCompute(Coefficients([1, 2, 3, 4, 5]))("example2", x=poly.y)

    cp = CoolPolynomialCompute([4, 42])
    cp.x.connect(23)

    m = ExternWithParams(8, 3)()
    m.name = "pexternInst"

    ports.y = poly.y


poly = pycde.System([PolynomialSystem], name="PolynomialSystem")
poly.print()

print("Generating 1...")
poly.generate(iters=1)

print("Printing...")
poly.print()
# CHECK-LABEL: msft.module @PolynomialSystem {} () -> (y: i32) {
# CHECK:         %example.y = msft.instance @example @PolyComputeForCoeff_62_42_6(%c23_i32) : (i32) -> i32
# CHECK:         %example2.y = msft.instance @example2 @PolyComputeForCoeff_62_42_6(%example.y) : (i32) -> i32
# CHECK:         %example2_1.y = msft.instance @example2_1 @PolyComputeForCoeff_1_2_3_4_5(%example.y) : (i32) -> i32
# CHECK:         %CoolPolynomialCompute.y = msft.instance @CoolPolynomialCompute @supercooldevice(%{{.+}}) : (i32) -> i32
# CHECK:         msft.instance @M @parameterized_extern() <a: i64 = 8, b: i64 = 3> : () -> ()
# CHECK:         msft.output %example.y : i32
# CHECK:       }
# CHECK:       msft.module @PolyComputeForCoeff_62_42_6 {coefficients = {coeff = [62, 42, 6]}} (%x: i32) -> (y: i32)
# CHECK:       msft.module @PolyComputeForCoeff_1_2_3_4_5 {coefficients = {coeff = [1, 2, 3, 4, 5]}} (%x: i32) -> (y: i32)
# CHECK:       msft.module.extern @supercooldevice(%x: i32) -> (y: i32) attributes {verilogName = "supercooldevice"}
# CHECK:       msft.module.extern @parameterized_extern<a: i64, b: i64>() attributes {verilogName = "parameterized_extern"}

print("Generating rest...")
poly.generate()

print("=== Post-generate IR...")
poly.run_passes()
poly.print()
# CHECK-LABEL: === Post-generate IR...
# CHECK: hw.module @PolynomialSystem
# CHECK: %example.y = hw.instance "example" sym @example @PolyComputeForCoeff_62_42_6(x: %c23_i32: i32) -> (y: i32)
# CHECK: %example2.y = hw.instance "example2" sym @example2 @PolyComputeForCoeff_62_42_6(x: %0: i32) -> (y: i32)
# CHECK: %example2_1.y = hw.instance "example2_1" sym @example2_1 @PolyComputeForCoeff_1_2_3_4_5(x: %1: i32) -> (y: i32)
# CHECK: %CoolPolynomialCompute.y = hw.instance "CoolPolynomialCompute" sym @CoolPolynomialCompute @supercooldevice(x: %c23_i32{{.*}}: i32) -> (y: i32)
# CHECK-LABEL: hw.module @PolyComputeForCoeff_62_42_6(%x: i32) -> (y: i32)
# CHECK: hw.constant 62
# CHECK: hw.constant 42
# CHECK: hw.constant 6
# CHECK-LABEL: hw.module @PolyComputeForCoeff_1_2_3_4_5(%x: i32) -> (y: i32)
# CHECK: hw.constant 1
# CHECK: hw.constant 2
# CHECK: hw.constant 3
# CHECK: hw.constant 4
# CHECK: hw.constant 5
# CHECK-NOT: hw.module @pycde.PolynomialCompute

poly.emit_outputs()

# OUTPUT-LABEL:   module PolyComputeForCoeff_62_42_6(
# OUTPUT:    input  [31:0] x,
# OUTPUT:    output [31:0] y);
