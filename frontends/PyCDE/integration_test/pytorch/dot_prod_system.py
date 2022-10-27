from pathlib import Path

from pycde import DefaultContext, Input, InputChannel, OutputChannel, esi, module, generator, types
from pycde.system import System

import torch
import torch_mlir


class DotModule(torch.nn.Module):

  def forward(self, a, b):
    return torch.matmul(a, b)


shape = torch_mlir.TensorPlaceholder([5], torch.int32)

torch_module = torch_mlir.compile(DotModule(), [shape, shape],
                                  output_type="linalg-on-tensors")
# with torch_module.context:
#   pm = torch_mlir.PassManager.parse(
#       """one-shot-bufferize{allow-return-allocs bufferize-function-boundaries},
#          buffer-results-to-out-params,
#          func.func(convert-linalg-to-affine-loops),
#          lower-affine,
#          convert-scf-to-cf,
#          canonicalize""")
#   pm.run(torch_module)

# print(torch_module)

# @esi.ServiceDecl
# class HandshakeServices:
#   go = esi.FromServer(types.i1)
#   done = esi.ToServer(types.i1)
#   read_mem = esi.ToFromServer(to_server_type=types.i64,
#                               to_client_type=types.i32)
#   result = esi.ToServer(types.i32)

# @module
# class DotProduct:
#   """An ESI-enabled module which only communicates with the host and computes
#   dot products."""
#   clock = Input(types.i1)
#   reset = Input(types.i1)

#   @generator
#   def generate(ports):
#     # Get the 'go' signal from the host.
#     go = HandshakeServices.go("dotprod_go")

#     # Instantiate the wrapped PyTorch dot product module.
#     wrapped_top = HandshakeToESIWrapper(clock=ports.clock,
#                                         reset=ports.reset,
#                                         go=go)

#     # Connect up the channels from the pytorch module.
#     HandshakeServices.done("dotprod_done", wrapped_top.done)
#     HandshakeServices.result("result", wrapped_top.result)

#     # Connect up the memory ports.
#     port0_data = HandshakeServices.read_mem("port0", wrapped_top.in0_ld_addr0)
#     wrapped_top.in0_ld_data0.connect(port0_data)
#     port1_data = HandshakeServices.read_mem("port1", wrapped_top.in1_ld_addr0)
#     wrapped_top.in1_ld_data0.connect(port1_data)


@module
class Top:
  clock = Input(types.i1)
  reset = Input(types.i1)

  @generator
  def generate(ports):
    pass
    # DotProduct(clock=ports.clock, reset=ports.reset)
    # esi.Cosim(HandshakeServices, ports.clock, ports.reset)


system = System([Top])
# system.import_mlir(torch_module)
system.generate()
system.emit_outputs()
