import torch
import torch_mlir


class DotModule(torch.nn.Module):

  def forward(self, a, b):
    return torch.matmul(a, b)


shape = torch_mlir.TensorPlaceholder([5], torch.int32)

module = torch_mlir.compile(DotModule(), [shape, shape],
                            output_type="linalg-on-tensors")

with open("dot.mlir", "w") as f:
  f.write(str(module))
