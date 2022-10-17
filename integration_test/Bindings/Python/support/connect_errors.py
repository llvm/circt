# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt
from circt.support import connect
from circt.dialects import hw


def build(top):
  i32 = mlir.ir.IntegerType.get_signless(32)
  dummy = hw.HWModuleOp(name='dummy',
                        input_ports=[('x', i32)],
                        output_ports=[('y', i32)],
                        body_builder=lambda mod: {'y': mod.x})
  const = hw.ConstantOp.create(i32, 0)
  inst = dummy.instantiate("dummy_inst", x=const.result)
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
