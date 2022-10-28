from pycde import Input, module, generator, types
from pycde.common import Clock
from pycde.system import System
from pycde.esi import FromServer, ServiceDecl

import torch
import torch_mlir


class DotModule(torch.nn.Module):

  def forward(self, a, b):
    return torch.matmul(a, b)


shape = torch_mlir.TensorPlaceholder([5], torch.int32)
torch_module = torch_mlir.compile(DotModule(), [shape, shape],
                                  output_type="linalg-on-tensors")


@ServiceDecl
class TorchControl:
  go = FromServer(types.i0)


@module
class Gasket:
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports):
    go = TorchControl.go()
    ForwardEsi(clock=ports.clk, reset=ports.rst, in3=go)
    # dot_a.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    # dot_b.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_x.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])


@module
class Top:
  clock = Clock()
  reset = Input(types.i1)

  @generator
  def generate(ports):
    Gasket(clk=ports.clock, rst=ports.reset)
    # esi.Cosim(HandshakeServices, ports.clock, ports.reset)


system = System([Top])
syms = system.import_mlir(torch_module)

ForwardEsi = syms["forward_esi_wrapper"]
dot_a: ServiceDecl = syms["in0"]
dot_b: ServiceDecl = syms["in1"]
dot_x: ServiceDecl = syms["in2"]

with open("dot_sys.pregen.mlir", "w") as f:
  f.write(str(system.mod))
system.generate()
with open("dot_sys.postgen.mlir", "w") as f:
  f.write(str(system.mod))
system.run_passes()
with open("dot_sys.postpasses.mlir", "w") as f:
  f.write(str(system.mod))
print("passed")
system.emit_outputs()
