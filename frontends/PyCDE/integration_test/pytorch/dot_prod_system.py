# REQUIRES: esi-cosim
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --tmpdir %t --schema %t/schema.capnp %s %t/*.sv
# PY: from dot_prod_system import run_cosim
# PY: run_cosim(tmpdir, rpcschemapath, simhostport)

from pycde import Input, module, generator, types
from pycde.common import Clock
from pycde.system import System
from pycde.esi import FromServer, ToFromServer, ServiceDecl, Cosim
from pycde.constructs import Wire

import torch
import torch_mlir
import numpy as np

import sys
import time
from typing import List


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
    go = TorchControl.go("go")
    ForwardEsi(clock=ports.clk, reset=ports.rst, in3=go)
    dot_a.write(TorchControl.a_write("a"))
    dot_b.write(TorchControl.b_write("b"))
    read_address = Wire(dot_x.read.to_server_type)
    read_data = dot_x.read(read_address)
    read_address.assign(TorchControl.x_read(read_data, "x"))
    dot_a.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_b.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_x.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])


@module
class top:
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports):
    Gasket(clk=ports.clk, rst=ports.rst)
    Cosim(TorchControl, ports.clk, ports.rst)


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
  system.emit_outputs()
  system.build_api("python")


def write_vector(vec: List[int], port):
  for i, v in enumerate(vec):
    port.write({"address": i, "data": v})


def hw_dotprod(hw, a: List[int], b: List[int]):
  write_vector(a, hw.torch_control.a_write[0])
  write_vector(b, hw.torch_control.b_write[0])
  hw.torch_control.go[0].write()
  time.sleep(0.01)
  x = hw.torch_control.x_read[0]()
  print(f"{a} x {b} = {x}")
  return x


def rand_vec():
  return [np.random.randint(0, 100) for _ in range(5)]


def run_cosim(tmpdir=".", schema_path="schema.capnp", rpchostport=None):
  sys.path.append(tmpdir)
  import esi_rt.PyTorchDotProd as esi_sys
  from esi_rt.common import Cosim
  if rpchostport is None:
    port = open("cosim.cfg").read().split(':')[1].strip()
    rpchostport = f"localhost:{port}"

  # Connect to RTL simulator via cosimulation.
  acc_conn = Cosim(schema_path, rpchostport)

  # Instantiate the accelerator host API with the backend connection.
  hw = esi_sys.top(acc_conn)

  # Run a simple dot product check.
  hw_dotprod(hw, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])

  # Instantiate PyTorch module for golden model.
  torch_dot = DotModule()
  for _ in range(25):
    a = rand_vec()
    b = rand_vec()

    # Compute with our accelerator.
    hdp = hw_dotprod(hw, a, b)

    # Compute known good result.
    tensor_a = torch.IntTensor(a)
    tensor_b = torch.IntTensor(b)
    swdp = torch_dot.forward(tensor_a, tensor_b)

    if hdp != swdp:
      print(f"  INCORRCT result. Correct is {swdp}")

  # import IPython
  # IPython.embed(colors="neutral")
