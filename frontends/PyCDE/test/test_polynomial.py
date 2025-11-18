# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file %t/hw/PolynomialCompute.sv --check-prefix=OUTPUT

from __future__ import annotations

import pycde
from pycde import (AppID, Input, Output, generator)
from pycde.module import Module, modparams
from pycde.dialects import comb, hw
from pycde.constructs import Wire
from pycde.types import Bit, Bits, StructType, TypeAlias

import sys


@modparams
def PolynomialCompute(coefficients: Coefficients):

  class PolynomialCompute(Module):
    """Module to compute ax^3 + bx^2 + cx + d for design-time coefficients"""
    module_name = f"PolyComputeForCoeff_{coefficients.coeff}"

    # Evaluate polynomial for 'x'.
    x = Input(Bits(32))
    y = Output(Bits(8 * 4))

    def __init__(self, name: str, **kwargs):
      """coefficients is in 'd' -> 'a' order."""
      super().__init__(instance_name=name, **kwargs)

    @generator
    def construct(mod):
      """Implement this module for input 'x'."""

      x = mod.x
      taps = list()
      for power, coeff in enumerate(coefficients.coeff):
        coeffVal = hw.ConstantOp(Bits(32), coeff)
        if power == 0:
          newPartialSum = coeffVal
        else:
          partialSum = taps[-1]
          if power == 1:
            currPow = x
          else:
            x_power = [x for i in range(power)]
            currPow = comb.MulOp(*x_power)
          newPartialSum = comb.AddOp(partialSum, comb.MulOp(coeffVal, currPow))

        taps.append(newPartialSum)

      # Final output
      mod.y = taps[-1]

  return PolynomialCompute


class CoolPolynomialCompute(Module):
  module_name = "supercooldevice"
  x = Input(Bits(32))
  y = Output(Bits(32))

  def __init__(self, coefficients, **inputs):
    super().__init__(**inputs)
    self.coefficients = coefficients


@modparams
def ExternWithParams(A: str, B: int):

  typedef1 = TypeAlias(StructType({"a": Bit}), "exTypedef")

  class M(Module):
    module_name = "parameterized_extern"
    ignored_input = Input(Bit)
    used_input = Input(Bits(B))

    @property
    def instance_name(self):
      return "singleton"

  return M


class Coefficients:

  def __init__(self, coeff):
    self.coeff = coeff


class PolynomialSystem(Module):
  y = Output(Bits(32))

  @generator
  def construct(self):
    i32 = Bits(32)
    x = hw.ConstantOp(i32, 23)
    poly = PolynomialCompute(Coefficients([62, 42, 6]))("example",
                                                        appid=AppID("poly"),
                                                        x=x)
    PolynomialCompute(coefficients=Coefficients([62, 42, 6]))("example2",
                                                              x=poly.y)
    PolynomialCompute(Coefficients([1, 2, 3, 4, 5]))("example2", x=poly.y)

    CoolPolynomialCompute([4, 42], x=23)

    w1 = Wire(Bits(4))
    m = ExternWithParams("foo", 4)(ignored_input=None, used_input=w1)
    m.name = "pexternInst"
    w1.assign(0)

    self._set_outputs(poly.outputs())


poly = pycde.System([PolynomialSystem],
                    name="PolynomialSystem",
                    output_directory=sys.argv[1])

# TODO: before generating all the modules, the IR doesn't verify since the
#       hw.instances don't resolve. Fix this.
# poly.print()

print("Generating rest...")
poly.generate()
poly.run_passes()

print("=== Final IR...")
poly.print()
# CHECK-LABEL: === Final IR...
# CHECK: hw.module @PolynomialSystem
# CHECK: %[[EXAMPLE_Y:.+]] = hw.instance "example" sym @example @PolyComputeForCoeff__62__42__6_(x: %c23_i32: i32) -> (y: i32)
# CHECK: %example2.y = hw.instance "example2" sym @example2 @PolyComputeForCoeff__62__42__6_(x: %[[EXAMPLE_Y]]: i32) -> (y: i32)
# CHECK: hw.instance "example2_1" sym @example2_1 @PolyComputeForCoeff__1__2__3__4__5_(x: %[[EXAMPLE_Y]]: i32)
# CHECK: %CoolPolynomialCompute.y = hw.instance "CoolPolynomialCompute" sym @CoolPolynomialCompute @supercooldevice(x: %c23_i32{{.*}}: i32) -> (y: i32)
# CHECK-LABEL: hw.module @PolyComputeForCoeff__62__42__6_(in %x : i32, out y : i32)
# CHECK-LABEL: hw.module @PolyComputeForCoeff__1__2__3__4__5_(in %x : i32, out y : i32)
# CHECK-NOT: hw.module @pycde.PolynomialCompute

poly.emit_outputs()

# OUTPUT-LABEL: `ifndef __PYCDE_TYPES__
# OUTPUT: `define __PYCDE_TYPES__
# OUTPUT: typedef struct packed {logic a; } exTypedef;
# OUTPUT: `endif // __PYCDE_TYPES__

# OUTPUT-LABEL:   module PolyComputeForCoeff__62__42__6_
# OUTPUT:    input  [31:0] x,
# OUTPUT:    output [31:0] y
# OUTPUT:    );
