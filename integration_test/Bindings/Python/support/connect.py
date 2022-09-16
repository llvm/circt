# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt
from circt.support import connect
from circt.dialects import hw


def build(mod, dummy_mod):
  i32 = mlir.ir.IntegerType.get_signless(32)
  # CHECK: %[[C0:.+]] = hw.constant 0
  const = hw.ConstantOp.create(i32, 0)
  inst = dummy_mod.instantiate("d")
  connect(inst.x, inst.y)
  connect(inst.x, const)
  connect(inst.x, const.result)
  # CHECK: hw.instance "d" @Dummy(x: %[[C0]]: i32)


with mlir.ir.Context() as ctx, mlir.ir.Location.unknown():
  circt.register_dialects(ctx)
  i32 = mlir.ir.IntegerType.get_signless(32)
  m = mlir.ir.Module.create()
  with mlir.ir.InsertionPoint(m.body):
    dummy = hw.HWModuleOp(name='Dummy',
                          input_ports=[("x", i32)],
                          output_ports=[("y", i32)],
                          body_builder=lambda m: {"y": m.x})

    hw.HWModuleOp(name='top',
                  input_ports=[],
                  output_ports=[],
                  body_builder=lambda top: build(top, dummy))
  print(m)
