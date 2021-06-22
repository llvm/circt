import circt.support as support
import circt.dialects.hw as hw


def connect(destination, source):
  if not isinstance(destination, support.OpOperand):
    raise TypeError(
        f"cannot connect to destination of type {type(destination)}")
  value = support.get_value(source)
  if value is None:
    raise TypeError(f"cannot connect from source of type {type(source)}")

  index = destination.index
  destination.operation.operands[index] = value
  if isinstance(destination, support.OpOperand) and \
     index in destination.builder.backedges:
    destination.builder.backedges[index].erase()


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
