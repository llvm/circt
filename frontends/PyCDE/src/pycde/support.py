import circt.support as support
import circt.dialects.hw as hw

import mlir.ir as ir

import os


class Value:

  def __init__(self, value, type):
    self.value = support.get_value(value)
    self.type = type

  def __getitem__(self, sub):
    if isinstance(self.type, hw.ArrayType):
      idx = int(sub)
      if idx >= self.type.size:
        raise ValueError("Subscript out-of-bounds")
      with get_user_loc():
        return hw.ArrayGetOp.create(self.value, idx)

    if isinstance(self.type, hw.StructType):
      with get_user_loc():
        return hw.StructExtractOp.create(self.value, sub)

    raise TypeError(
        "Subscripting only supported on hw.array and hw.struct types")


# PyCDE needs a custom version of this to support python classes.
def var_to_attribute(obj) -> ir.Attribute:
  """Create an MLIR attribute from a Python object for a few common cases."""
  if obj is None:
    return ir.BoolAttr.get(False)
  if isinstance(obj, ir.Attribute):
    return obj
  if isinstance(obj, bool):
    return ir.BoolAttr.get(obj)
  if isinstance(obj, int):
    attrTy = ir.IntegerType.get_signless(64)
    return ir.IntegerAttr.get(attrTy, obj)
  if isinstance(obj, str):
    return ir.StringAttr.get(obj)
  if isinstance(obj, list) or isinstance(obj, tuple):
    arr = [var_to_attribute(x) for x in obj]
    if all(arr):
      return ir.ArrayAttr.get(arr)
  if isinstance(obj, dict):
    attrs = {name: var_to_attribute(value) for name, value in obj.items()}
    return ir.DictAttr.get(attrs)
  if hasattr(obj, "__dict__"):
    attrs = {name: var_to_attribute(value)
             for name, value in obj.__dict__.items()}
    return ir.DictAttr.get(attrs)
  raise TypeError(f"Cannot convert type '{type(obj)}' to MLIR attribute. "
                  "This is required for parameters.")


__dir__ = os.path.dirname(__file__)
_local_files = set([os.path.join(__dir__, x) for x in os.listdir(__dir__)])


def get_user_loc() -> ir.Location:
  import traceback
  stack = reversed(traceback.extract_stack())
  for frame in stack:
    if frame.filename in _local_files:
      continue
    return ir.Location.file(frame.filename, frame.lineno, 0)
  return ir.Location.unknown()
