# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt

from circt.dialects import hw
from circt.esi import types
from circt.support import BackedgeBuilder


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
    const = hw.ConstantOp(types.i32, mlir.ir.IntegerAttr.get(types.i32, 0))
    dummy = Dummy()
    inst = dummy.module.create("d", {"x": const.result})
    try:
        # CHECK: cannot connect from source of type
        circt.connect(inst.y, None)
    except TypeError as e:
        print(e)
    try:
        # CHECK: cannot connect to destination of type
        circt.connect(None, inst.x)
    except TypeError as e:
        print(e)


with mlir.ir.Context() as ctxt, mlir.ir.Location.unknown():
  circt.register_dialects(ctxt)
  m = mlir.ir.Module.create()
  with mlir.ir.InsertionPoint(m.body):
    Test()
