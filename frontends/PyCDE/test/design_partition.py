# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

import pycde
from pycde import (DesignPartition, Input, Output, module, externmodule,
                   generator, types)

import sys


@module
class TopLevel:
  x = Input(types.i32)
  y = Output(types.i32)

  @generator
  def construct(mod):
    TopLevel.part1 = DesignPartition("part1")
    Plus("Plus1", a=mod.x, b=mod.x, partition=TopLevel.part1)
    mod.y = PlusPipeline(a=mod.x).y


@externmodule
class Plus:
  a = Input(types.i32)
  b = Input(types.i32)
  y = Output(types.i32)

  def __init__(self, name: str = None) -> None:
    if name is not None:
      self.instance_name = name


@module
class PlusPipeline:
  a = Input(types.i32)
  y = Output(types.i32)

  @generator
  def construct(mod):
    p1 = Plus(a=mod.a, b=mod.a)
    p2 = Plus(a=p1.y, b=mod.a, partition=TopLevel.part1)
    p3 = Plus(a=p2.y, b=mod.a)
    mod.y = p3.y


s = pycde.System([TopLevel],
                 name="DesignPartitionTest",
                 output_directory=sys.argv[1])

print("Generating...")
s.generate()

s.print()
s.emit_outputs()
print("************")
print("** Post pass/emit")
s.print()

# CHECK-LABEL: msft.module @TopLevel
# CHECK:         msft.partition @part1, "part1"
# CHECK:         %Plus1.y = msft.instance @Plus1 @Plus(%x, %x)  {targetDesignPartition = @TopLevel::@part1} : (i32, i32) -> i32
# CHECK:         %PlusPipeline.y = msft.instance @PlusPipeline @PlusPipeline(%x)  : (i32) -> i32
# CHECK:         msft.output %PlusPipeline.y : i32

# CHECK-LABEL: ** Post pass/emit
# CHECK-LABEL: hw.module @TopLevel
# CHECK:         %part1_Plus1_y = sv.wire  : !hw.inout<i32>
# CHECK:         %part1.Plus1_y = hw.instance "part1" sym @part1 @part1(Plus1_a: %x: i32, Plus1_b: %x: i32) -> (Plus1_y: i32)
# CHECK:         sv.assign %part1_Plus1_y, %part1.Plus1_y : i32
# CHECK:         %PlusPipeline.y = hw.instance "PlusPipeline" sym @PlusPipeline @PlusPipeline(a: %x: i32) -> (y: i32)
# CHECK-LABEL: hw.module @part1(%Plus1_a: i32, %Plus1_b: i32) -> (Plus1_y: i32) {
# CHECK:         %Plus1.y = hw.instance "Plus1" sym @Plus1 @Plus(a: %Plus1_b: i32, b: %Plus1_b: i32) -> (y: i32)
# CHECK:         hw.output %Plus1.y : i32
