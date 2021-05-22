# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt

from circt.design_entry import module, Input, Output, connect
from circt.dialects import hw
from circt.esi import types


@module
class Dummy:

  def __init__(self):
    self.x = Input(types.i32)
    self.y = Output(types.i32)

  def construct(self, x):
    self.y.set(x)


@module
class Test:

  def construct(self):
    # Temporarily broken: %[[C0:.+]] = hw.constant 0
    const = hw.ConstantOp(types.i32, mlir.ir.IntegerAttr.get(types.i32, 0))
    dummy = Dummy()
    inst = dummy.module.create("d")
    connect(inst.x, inst.y)
    connect(inst.x, const)
    connect(inst.x, const.result)
    # Temporarily broken: hw.instance "d" @Dummy(%[[C0]])


with mlir.ir.Context() as ctxt, mlir.ir.Location.unknown():
  circt.register_dialects(ctxt)
  m = mlir.ir.Module.create()
  with mlir.ir.InsertionPoint(m.body):
    Test()
    # CHECK:  "circt.design_entry.Test"() : () -> ()
  print(m)
