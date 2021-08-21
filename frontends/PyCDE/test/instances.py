# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import pycde
import circt.dialects.hw


@pycde.module
class Nothing:

  @pycde.generator
  def construct(mod):
    return {}


@pycde.module
class UnParameterized:
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(mod):
    Nothing()
    return {"y": mod.x}


@pycde.module
class Test:
  inputs = []
  outputs = []

  @pycde.generator
  def build(_):
    c1 = circt.dialects.hw.ConstantOp.create(pycde.types.i1, 1)
    UnParameterized(x=c1)
    UnParameterized(x=c1)


# CHECK: hw.module @pycde.UnParameterized
# CHECK-NOT: hw.module @pycde.UnParameterized
t = pycde.System([Test])
t.generate(["construct"])
t.print()

mod = t.get_module("Test")
print(mod)

t.run_passes()

# CHECK-LABEL: === Hierarchy
print("=== Hierarchy")
# CHECK-NEXT: <instance: [pycde_UnParameterized]>
# CHECK-NEXT: <instance: [pycde_UnParameterized, pycde_Nothing]>
# CHECK-NEXT: <instance: [pycde_UnParameterized_0]>
# CHECK-NEXT: <instance: [pycde_UnParameterized_0, pycde_Nothing]>
t.walk_instances("Test", lambda inst: print(inst))
