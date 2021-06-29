import circt.support as support
import circt.dialects.hw as hw

import mlir.ir


class Value:

  def __init__(self, value, type):
    self.value = support.get_value(value)
    self.type = type

  def __getitem__(self, sub):
    if isinstance(self.type, hw.ArrayType):
      idx = int(sub)
      if idx >= self.type.size:
        raise ValueError("Subscript out-of-bounds")
      return hw.ArrayGetOp.create(self.value, idx)

    if isinstance(self.type, hw.StructType):
      return hw.StructExtractOp.create(self.value, sub)

    raise TypeError(
        "Subscripting only supported on hw.array and hw.struct types")
