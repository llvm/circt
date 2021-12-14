# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file %t/TopLevel.sv --check-prefix=OUTPUT

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
    mod.y = PlusPipeline(a=mod.x).y


@externmodule
class Plus:
  a = Input(types.i32)
  b = Input(types.i32)
  y = Output(types.i32)


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
