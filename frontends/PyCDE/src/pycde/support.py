import circt.support as support
from circt.dialects import hw

import mlir.ir as ir

import os


# PyCDE needs a custom version of this to support python classes.
def var_to_attribute(obj) -> ir.Attribute:
  """Create an MLIR attribute from a Python object for a few common cases."""
  if obj is None:
    return ir.BoolAttr.get(False)
  if isinstance(obj, ir.Attribute):
    return obj
  if isinstance(obj, ir.Type):
    return ir.TypeAttr.get(obj)
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
    attrs = {
        name: var_to_attribute(value) for name, value in obj.__dict__.items()
    }
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


def create_const_zero(type: ir.Type):
  """Create a 'default' constant value of zero. Used for creating dummy values
  to connect to extern modules with input ports we want to ignore."""
  width = hw.get_bitwidth(type)
  zero = hw.ConstantOp.create(ir.IntegerType.get_signless(width), 0)
  return hw.BitcastOp(type, zero.result)


class OpOperandConnect(support.OpOperand):
  """An OpOperand pycde extension which adds a connect method."""

  def connect(self, obj, result_type=None):
    val = obj_to_value(obj, self.type, result_type)
    support.connect(self, val)


def obj_to_value(x, type, result_type=None):
  """Convert a python object to a CIRCT value, given the CIRCT type."""
  assert x is not None
  from .value import Value

  type = support.type_to_pytype(type)
  if isinstance(type, hw.TypeAliasType):
    return obj_to_value(x, type.inner_type, type)

  if result_type is None:
    result_type = type
  else:
    result_type = support.type_to_pytype(result_type)
    assert isinstance(result_type, hw.TypeAliasType) or result_type == type

  val = support.get_value(x)
  # If x is already a valid value, just return it.
  if val is not None:
    if val.type != result_type:
      raise ValueError(f"Expected {result_type}, got {val.type}")
    return Value.get(val)

  if isinstance(x, int):
    if not isinstance(type, ir.IntegerType):
      raise ValueError(f"Int can only be converted to hw int, not '{type}'")
    return Value.get(hw.ConstantOp.create(type, x).result)

  if isinstance(x, list):
    if not isinstance(type, hw.ArrayType):
      raise ValueError(f"List is only convertable to hw array, not '{type}'")
    elemty = type.element_type
    if len(x) != type.size:
      raise ValueError("List must have same size as array "
                       f"{len(x)} vs {type.size}")
    list_of_vals = list(map(lambda x: obj_to_value(x, elemty), x))
    # CIRCT's ArrayCreate op takes the array in reverse order.
    return Value.get(hw.ArrayCreateOp.create(reversed(list_of_vals)).result)

  if isinstance(x, dict):
    if not isinstance(type, hw.StructType):
      raise ValueError(f"Dict is only convertable to hw struct, not '{type}'")
    fields = type.get_fields()
    elem_name_values = []
    for (fname, ftype) in fields:
      if fname not in x:
        raise ValueError(f"Could not find expected field: {fname}")
      elem_name_values.append((fname, obj_to_value(x[fname], ftype)))
      x.pop(fname)
    if len(x) > 0:
      raise ValueError(f"Extra fields specified: {x}")
    return Value.get(
        hw.StructCreateOp.create(elem_name_values,
                                 result_type=result_type).result)

  raise ValueError(f"Unable to map object '{type(x)}' to MLIR Value")


def create_type_string(ty):
  ty = support.type_to_pytype(ty)
  if isinstance(ty, hw.TypeAliasType):
    return ty.name
  if isinstance(ty, hw.ArrayType):
    return f"{ty.size}x" + create_type_string(ty.element_type)
  return str(ty)
