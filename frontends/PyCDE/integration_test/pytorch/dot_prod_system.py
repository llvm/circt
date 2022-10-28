# REQUIRES: esi-cosim
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# : esi-cosim-runner.py --tmpdir %t --schema %t/schema.capnp %s %t/*.sv
# PY: from dot_prod_system import run_cosim
# PY: run_cosim(tmpdir, rpcschemapath, simhostport)

from pycde import Input, module, generator, types
from pycde.common import Clock
from pycde.system import System
from pycde.esi import FromServer, ToFromServer, ServiceDecl, Cosim
from pycde.constructs import Wire

import torch
import torch_mlir

import sys


class DotModule(torch.nn.Module):

  def forward(self, a, b):
    return torch.matmul(a, b)


shape = torch_mlir.TensorPlaceholder([5], torch.int32)
torch_module = torch_mlir.compile(DotModule(), [shape, shape],
                                  output_type="linalg-on-tensors")


@module
class Gasket:
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports):
    go = TorchControl.go()
    ForwardEsi(clock=ports.clk, reset=ports.rst, in3=go)
    dot_a.write(TorchControl.a_write())
    dot_b.write(TorchControl.b_write())
    read_address = Wire(dot_x.read.to_server_type)
    read_data = dot_x.read(read_address)
    read_address.assign(TorchControl.x_read(read_data))
    dot_a.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_b.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_x.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])


@module
class top:
  clock = Clock()
  reset = Input(types.i1)

  @generator
  def generate(ports):
    Gasket(clk=ports.clock, rst=ports.reset)
    Cosim(TorchControl, ports.clock, ports.reset)


if __name__ == "__main__":
  system = System([top], name="PyTorchDotProd", output_directory=sys.argv[1])
  syms = system.import_mlir(torch_module)

  ForwardEsi = syms["forward_esi_wrapper"]
  dot_a: ServiceDecl = syms["in0"]
  dot_b: ServiceDecl = syms["in1"]
  dot_x: ServiceDecl = syms["in2"]

  @ServiceDecl
  class TorchControl:
    go = FromServer(types.i0)
    a_write = FromServer(dot_a.write.to_server_type)
    b_write = FromServer(dot_b.write.to_server_type)
    x_read = ToFromServer(dot_x.read.to_client_type, dot_x.read.to_server_type)

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


def run_cosim(tmpdir=".", schema_path="schema.capnp", rpchostport=None):
  sys.path.append(tmpdir)
  import esi_rt.PyTorchDotProd as esi_sys
  from esi_rt.common import Cosim
  if rpchostport is None:
    port = open("cosim.cfg").read().split(':')[1].strip()
    rpchostport = f"localhost:{port}"

  cosim = Cosim(schema_path, rpchostport)
  print(cosim.list())
  top = esi_sys.top(cosim)
