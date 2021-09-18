# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import pycde
import circt.dialects.hw


@pycde.module
def Parameterized(param):

  class TestModule:
    x = pycde.Input(pycde.types.i1)
    y = pycde.Output(pycde.types.i1)

    @pycde.generator
    def construct(mod):
      return {"y": mod.x}

  return TestModule


@pycde.module
class UnParameterized:
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(mod):
    return {"y": mod.x}


@pycde.module
class Test:
  inputs = []
  outputs = []

  @pycde.generator
  def build(_):
    c1 = circt.dialects.hw.ConstantOp.create(pycde.types.i1, 1)
    Parameterized(1)(x=c1)
    Parameterized(1)(x=c1)
    Parameterized(2)(x=c1)
    Parameterized(2)(x=c1)
    UnParameterized(x=c1)
    UnParameterized(x=c1)


# CHECK: hw.module @pycde.Module_1
# CHECK-NOT: hw.module @pycde.Module_1
# CHECK: hw.module @pycde.Module_2
# CHECK-NOT: hw.module @pycde.Module_2
# CHECK: hw.module @pycde.UnParameterized
# CHECK-NOT: hw.module @pycde.UnParameterized
t = pycde.System([Test])
t.generate()
t.generate(["construct"])
t.print()
