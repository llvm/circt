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


class Test(pycde.System):
  inputs = []
  outputs = []

  def build(self, top):
    c1 = circt.dialects.hw.ConstantOp.create(pycde.types.i1, 1)
    Parameterized(1)(x=c1)
    Parameterized(1)(x=c1)
    Parameterized(2)(x=c1)
    Parameterized(2)(x=c1)


# CHECK: hw.module @Module
# CHECK-NOT: hw.module @Module
# CHECK: hw.module @Module_1
# CHECK-NOT: hw.module @Module_1
# CHECK: hw.module @Module_2
# CHECK-NOT: hw.module @Module_2
t = Test()
t.generate()
t.print()
