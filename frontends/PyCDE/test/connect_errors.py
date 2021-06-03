# RUN: %PYTHON% %s | FileCheck %s

import mlir

from pycde import module, generator, Input, Output
from circt.support import connect
from circt.dialects import hw
from circt.esi import types


@module
class Dummy:

  def __init__(self, **kwargs):
    self.x = Input(types.i32)
    self.y = Output(types.i32)

  @generator
  def construct(mod):
    return {'y': mod.x}


@module
class Test:

  @generator
  def construct(mod):
    const = hw.ConstantOp.create(types.i32, 0)
    dummy = Dummy(x=const)
    try:
      # CHECK: cannot connect from source of type
      connect(dummy.x, None)
    except TypeError as e:
      print(e)
    try:
      # CHECK: cannot connect to destination of type
      connect(None, dummy.x)
    except TypeError as e:
      print(e)


def build(top):
  Test()


mod = mlir.ir.Module.create()
with mlir.ir.InsertionPoint(mod.body):
  hw.HWModuleOp(name='top',
                input_ports=[],
                output_ports=[],
                body_builder=build)

pm = mlir.passmanager.PassManager.parse("run-generators")
pm.run(mod)
