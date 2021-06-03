# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt
from circt.support import connect
from circt.dialects import hw
from circt.esi import types


def build(top):
  dummy = hw.HWModuleOp(name='dummy',
                        input_ports=[('x', types.i32)],
                        output_ports=[('y', types.i32)],
                        body_builder=lambda mod: {'y': mod.x})
  const = hw.ConstantOp.create(types.i32, 0)
  inst = dummy.create("dummy_inst", x=const.result)
  try:
    # CHECK: cannot connect from source of type
    connect(inst.x, None)
  except TypeError as e:
    print(e)
  try:
    # CHECK: cannot connect to destination of type
    connect(None, inst.x)
  except TypeError as e:
    print(e)


with mlir.ir.Context() as ctx, mlir.ir.Location.unknown():
  circt.register_dialects(ctx)

  mod = mlir.ir.Module.create()
  with mlir.ir.InsertionPoint(mod.body):
    hw.HWModuleOp(name='top',
                  input_ports=[],
                  output_ports=[],
                  body_builder=build)
