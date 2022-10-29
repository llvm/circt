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

################################################################################
# Harware design
################################################################################


class DotModule(torch.nn.Module):

  def forward(self, a, b):
    return torch.matmul(a, b)


shape = torch_mlir.TensorPlaceholder([5], torch.int32)
torch_module = torch_mlir.compile(DotModule(), [shape, shape],
                                  output_type="linalg-on-tensors")


@module
class top:
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports):
    Gasket(clk=ports.clk, rst=ports.rst)

    # Use cosim to communicate with the accelerator running in an RTL simulator.
    Cosim(TorchControl, ports.clk, ports.rst)


@module
class Gasket:
  """Wrap the accelerator IP module. Instantiate the requiste memories. Wire the
  memories to the host and the host to the module control signals."""

  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports):
    # Get a 'go' signal from the host.
    go = TorchControl.go("go")

    # Instantiate the accelerator IP, passing it the 'go' signal from the host.
    # All other communication is done through ESI services.
    DotAccelIP(clock=ports.clk, reset=ports.rst, in3=go)

    # Implement the three memories which the IP needs with simple SystemVerilog
    # unpacked arrays.
    dot_a.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_b.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_x.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])

    # Give the host access to the memories. Write access to the 'a' and 'b'
    # memories and read access to the 'x' memory.
    dot_a.write(TorchControl.a_write("a"))
    dot_b.write(TorchControl.b_write("b"))
    read_address = Wire(dot_x.read.to_server_type)
    read_data = dot_x.read(read_address)
    read_address.assign(TorchControl.x_read(read_data, "x"))


if __name__ == "__main__":
  system = System([top], name="PyTorchDotProd", output_directory=sys.argv[1])

  # Import the torch_mlir module.
  syms = system.import_mlir(torch_module)

  # Grab references to the imported IP and requested memories which must
  # implemented in a gasket/wrapper.
  DotAccelIP = syms["forward_esi_wrapper"]
  dot_a: ServiceDecl = syms["in0"]
  dot_b: ServiceDecl = syms["in1"]
  dot_x: ServiceDecl = syms["in2"]

  # Define an interface (API) for software.
  @ServiceDecl
  class TorchControl:
    go = FromServer(types.i0)
    a_write = FromServer(dot_a.write.to_server_type)
    b_write = FromServer(dot_b.write.to_server_type)
    x_read = ToFromServer(dot_x.read.to_client_type, dot_x.read.to_server_type)

  # Generate the hardware design.
  system.generate()
  # Emit SystemVerilog and other collateral.
  system.emit_outputs()
  # Build a Python API into the accelerator.
  system.build_api("python")

################################################################################
# Software runtime
################################################################################


def write_vector(vec: List[int], port):
  for i, v in enumerate(vec):
    port.write({"address": i, "data": v})


def hw_dotprod(hw, a: List[int], b: List[int]):
  # Write the two vectors to device memories.
  write_vector(a, hw.torch_control.a_write[0])
  write_vector(b, hw.torch_control.b_write[0])

  # Tell the dot module to go!
  hw.torch_control.go[0].write()

  # Wait for unit to compute.
  #       (Hack around missing functionality in XRT bridge.)
  time.sleep(0.01)

  # Read the result.
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
