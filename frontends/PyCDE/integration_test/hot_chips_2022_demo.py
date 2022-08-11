from pathlib import Path

from mlir.ir import Module, NoneType

from circt.dialects import hw

from pycde import InputChannel, OutputChannel, module, generator, types
from pycde.system import System
from pycde.module import import_hw_module

path = Path(__file__).parent / "hot_chips_2022_example.mlir"
mlir_module = Module.parse(open(path).read())
imported_modules = []
top = None
for op in mlir_module.body:
  if isinstance(op, hw.HWModuleOp):
    imported_module = import_hw_module(op)
    imported_modules.append(imported_module)
    if imported_module._pycde_mod.name == 'forward':
      top = imported_module

@module
class HandshakeToESIWrapper:
  # Control Ports

  ## Generic ports always present
  clock = types.i1
  reset = types.i1

  ## Go signal
  inCtrl = InputChannel(NoneType.get())

  ## Done signal
  outCtrl = OutputChannel(NoneType.get())

  # Input 0 Ports

  ## Channels from Memory
  in0_ldData0 = InputChannel(types.i32)

  in0_ldDone0 = InputChannel(NoneType.get())

  ## Channels to Memory
  in0_ldAddr0 = OutputChannel(types.i64)

  # Input 1 Ports

  ## Channels from Memory
  in1_ldData0 = InputChannel(types.i32)

  in1_ldDone0 = InputChannel(NoneType.get())

  ## Channels to Memory
  in1_ldAddr0 = OutputChannel(types.i64)

  # Output 0 Ports

  ## Channels to Host
  out0 = OutputChannel(types.i32)

  @generator
  def generate(mod):
    top()

system = System([HandshakeToESIWrapper])
system.import_modules(imported_modules)
system.generate()
# system.print()
