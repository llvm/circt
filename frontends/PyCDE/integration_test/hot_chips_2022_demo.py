from pathlib import Path

from mlir.ir import Module, NoneType

from circt.dialects import hw

from pycde import Input, InputChannel, Output, OutputChannel, esi, module, generator, types
from pycde.dialects import comb
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

  ## Channels to Memory
  in0_ld_addr0 = OutputChannel(types.i64)

  # Input 1 Ports

  ## Channels from Memory
  in1_ld_data0 = InputChannel(types.i32)

  ## Channels to Memory
  in1_ld_addr0 = OutputChannel(types.i64)

  # Output 0 Ports

  ## Channels to Host
  out0 = OutputChannel(types.i32)

  @generator
  def generate(ports):
    none_channel = types.channel(NoneType.get())
    i32_channel = types.channel(types.i32)
    i64_channel = types.channel(types.i64)

    # Instantiate the top-level module to wrap with backedges for most ports.
    wrapped_top = top(clock=ports.clock, reset=ports.reset)

    # Control Ports

    ## Go signal
    _, in_ctrl_valid = ports.in_ctrl.unwrap(wrapped_top.inCtrl_ready)
    wrapped_top.inCtrl_valid.connect(in_ctrl_valid)

    ## Done signal
    out_ctrl_channel, out_ctrl_ready = none_channel.wrap(None, wrapped_top.outCtrl_valid)
    wrapped_top.outCtrl_ready.connect(out_ctrl_ready)
    ports.out_ctrl = out_ctrl_channel

    # Input 0 Ports

    ## Channels from Memory
    in0_ready = comb.AndOp(wrapped_top.in0_ldData0_ready, wrapped_top.in0_ldDone0_ready)

    in0_ld_data0_data, in0_ld_data0_valid = ports.in0_ld_data0.unwrap(in0_ready)
    wrapped_top.in0_ldData0_data.connect(in0_ld_data0_data)
    wrapped_top.in0_ldData0_valid.connect(in0_ld_data0_valid)
    wrapped_top.in0_ldDone0_valid.connect(in0_ld_data0_valid)

    ## Channels to Memory
    in0_ld_addr0_channel, in0_ld_addr0_ready = i64_channel.wrap(wrapped_top.in0_ldAddr0_data, wrapped_top.in0_ldAddr0_valid)
    wrapped_top.in0_ldAddr0_ready.connect(in0_ld_addr0_ready)
    ports.in0_ld_addr0 = in0_ld_addr0_channel

    # Input 1 Ports

    ## Channels from Memory
    in1_ready = comb.AndOp(wrapped_top.in1_ldData0_ready, wrapped_top.in1_ldDone0_ready)

    in1_ld_data0_data, in1_ld_data0_valid = ports.in1_ld_data0.unwrap(in1_ready)
    wrapped_top.in1_ldData0_data.connect(in1_ld_data0_data)
    wrapped_top.in1_ldData0_valid.connect(in1_ld_data0_valid)
    wrapped_top.in1_ldDone0_valid.connect(in1_ld_data0_valid)

    ## Channels to Memory
    in1_ld_addr0_channel, in1_ld_addr0_ready = i64_channel.wrap(wrapped_top.in1_ldAddr0_data, wrapped_top.in1_ldAddr0_valid)
    wrapped_top.in1_ldAddr0_ready.connect(in1_ld_addr0_ready)
    ports.in1_ld_addr0 = in1_ld_addr0_channel

    # Output 0 Ports
    out0_channel, out0_ready = i32_channel.wrap(wrapped_top.out0_data, wrapped_top.out0_valid)
    wrapped_top.out0_ready.connect(out0_ready)
    ports.out0 = out0_channel


@esi.ServiceDecl
class Control:
  ctrl = esi.ToFromServer(to_server_type=NoneType.get(), to_client_type=NoneType.get())


@esi.ServiceDecl
class Memory:
  port0 = esi.ToFromServer(to_server_type=types.i64, to_client_type=types.i32)
  port1 = esi.ToFromServer(to_server_type=types.i64, to_client_type=types.i32)


@esi.ServiceDecl
class Result:
  data = esi.ToServer(types.i32)


@module
class ServiceWrapper:
  clock = Input(types.i1)
  reset = Input(types.i1)

  @generator
  def generate(ports):
    wrapped_top = HandshakeToESIWrapper(clock=ports.clock, reset=ports.reset)

    ctrl = Control.ctrl("ctrl", wrapped_top.out_ctrl)
    wrapped_top.in_ctrl.connect(ctrl)

    port0_data = Memory.port0("port0", wrapped_top.in0_ld_addr0)
    wrapped_top.in0_ld_data0.connect(port0_data)

    port1_data = Memory.port1("port1", wrapped_top.in1_ld_addr0)
    wrapped_top.in1_ld_data0.connect(port1_data)

    Result.data("data", wrapped_top.out0)

@module
class Top:
  clock = Input(types.i1)
  reset = Input(types.i1)

  @generator
  def generate(ports):
    ServiceWrapper(clock=ports.clock, reset=ports.reset)
    esi.Cosim(Memory, ports.clock, ports.reset)

system = System([Top])
system.import_modules(imported_modules)
system.generate()
system.emit_outputs()
