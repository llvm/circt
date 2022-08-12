from pathlib import Path

from mlir.ir import Module, NoneType

from circt.dialects import hw

from pycde import Input, InputChannel, Output, OutputChannel, module, generator, types
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
  clock = Input(types.i1)
  reset = Input(types.i1)

  ## Go signal
  in_ctrl = InputChannel(NoneType.get())

  ## Done signal
  out_ctrl = OutputChannel(NoneType.get())

  # Input 0 Ports

  ## Channels from Memory
  in0_ld_data0 = InputChannel(types.i32)

  in0_ld_done0 = InputChannel(NoneType.get())

  ## Channels to Memory
  in0_ld_addr0 = OutputChannel(types.i64)

  # Input 1 Ports

  ## Channels from Memory
  in1_ld_data0 = InputChannel(types.i32)

  in1_ld_done0 = InputChannel(NoneType.get())

  ## Channels to Memory
  in1_ld_addr0 = OutputChannel(types.i64)

  # Output 0 Ports

  ## Channels to Host
  out0 = OutputChannel(types.i32)

  @generator
  def generate(ports):
    wrapped_top = top(clock=ports.clock, reset=ports.reset)

    _, in_ctrl_valid = ports.in_ctrl.unwrap(wrapped_top.inCtrl_ready)
    wrapped_top.inCtrl_valid.connect(in_ctrl_valid)


system = System([HandshakeToESIWrapper])
system.import_modules(imported_modules)
system.generate()
# system.print()
