# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import pycde
import circt.dialects.hw


@pycde.module
def Parameterized(param):

  class Module:
    x = pycde.Input(pycde.types.i1)
    y = pycde.Output(pycde.types.i1)

    @pycde.generator
    def construct(mod):
      return {"y": mod.x}

  return Module


@pycde.module
class UnParameterized:
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(mod):
    return {"y": mod.x}


class Test(pycde.System):
  inputs = []
  outputs = []

  def build(self, top):
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
t = Test()
t.generate(["construct"])
t.print()
