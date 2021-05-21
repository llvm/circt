# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt

from circt.dialects import hw
from circt.esi import types


@circt.module
class Dummy:

  def __init__(self):
    self.x = circt.Input(types.i32)
    self.y = circt.Output(types.i32)

  def construct(self, x):
    self.y.set(x)


@circt.module
class Test:

  def construct(self):
    # CHECK: %[[C0:.+]] = hw.constant 0
    const = hw.ConstantOp(types.i32, mlir.ir.IntegerAttr.get(types.i32, 0))
    dummy = Dummy()
    inst = dummy.module.create("d")
    circt.connect(inst.x, inst.y)
    circt.connect(inst.x, const)
    circt.connect(inst.x, const.result)
    # CHECK: hw.instance "d" @Dummy(%[[C0]])


with mlir.ir.Context() as ctxt, mlir.ir.Location.unknown():
  circt.register_dialects(ctxt)
  m = mlir.ir.Module.create()
  with mlir.ir.InsertionPoint(m.body):
    Test()
  print(m)
