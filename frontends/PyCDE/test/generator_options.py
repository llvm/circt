# RUN: %PYTHON% %s | FileCheck %s

from pycde import System, module, generator, types

from circt.dialects import hw


@module
class GeneratorOptions:

  @generator
  def generator_a(mod):
    hw.ConstantOp.create(types.i32, 1)

  @generator
  def generator_b(mod):
    hw.ConstantOp.create(types.i32, 2)


class Top(System):
  inputs = []
  outputs = []

  def build(self, top):
    GeneratorOptions()


# CHECK: hw.constant 1
top1 = Top()
top1.generate()
top1.print()

# CHECK: hw.constant 2
top2 = Top()
top2.generate(["generator_b"])
top2.print()

# CHECK: generator exception
top3 = Top()
try:
  top3.generate(["generator_a", "generator_b"])
except RuntimeError:
  pass
