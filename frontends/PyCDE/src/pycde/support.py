import circt.support as support

import mlir.ir as ir

import os


# PyCDE needs a custom version of this to support python classes.
def _obj_to_attribute(obj) -> ir.Attribute:
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
    arr = [_obj_to_attribute(x) for x in obj]
    if all(arr):
      return ir.ArrayAttr.get(arr)
  if isinstance(obj, dict):
    attrs = {name: _obj_to_attribute(value) for name, value in obj.items()}
    return ir.DictAttr.get(attrs)
  if hasattr(obj, "__dict__"):
    attrs = {
        name: _obj_to_attribute(value) for name, value in obj.__dict__.items()
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
  from .dialects import hw
  width = hw.get_bitwidth(type)

  with get_user_loc():
    zero = hw.ConstantOp(ir.IntegerType.get_signless(width), 0)
    return hw.BitcastOp(type, zero)


class OpOperandConnect(support.OpOperand):
  """An OpOperand pycde extension which adds a connect method."""

  def connect(self, obj, result_type=None):
    val = _obj_to_value(obj, self.type, result_type)
    support.connect(self, val)


def _obj_to_value(x, type, result_type=None):
  """Convert a python object to a CIRCT value, given the CIRCT type."""
  assert x is not None
  from .value import Value
  from .dialects import hw

  type = support.type_to_pytype(type)
  if isinstance(type, hw.TypeAliasType):
    return _obj_to_value(x, type.inner_type, type)

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
    with get_user_loc():
      return hw.ConstantOp(type, x)

  if isinstance(x, list):
    if not isinstance(type, hw.ArrayType):
      raise ValueError(f"List is only convertable to hw array, not '{type}'")
    elemty = type.element_type
    if len(x) != type.size:
      raise ValueError("List must have same size as array "
                       f"{len(x)} vs {type.size}")
    list_of_vals = list(map(lambda x: _obj_to_value(x, elemty), x))
    # CIRCT's ArrayCreate op takes the array in reverse order.
    with get_user_loc():
      return hw.ArrayCreateOp(reversed(list_of_vals))

  if isinstance(x, dict):
    if not isinstance(type, hw.StructType):
      raise ValueError(f"Dict is only convertable to hw struct, not '{type}'")
    fields = type.get_fields()
    elem_name_values = []
    for (fname, ftype) in fields:
      if fname not in x:
        raise ValueError(f"Could not find expected field: {fname}")
      elem_name_values.append((fname, _obj_to_value(x[fname], ftype)))
      x.pop(fname)
    if len(x) > 0:
      raise ValueError(f"Extra fields specified: {x}")
    with get_user_loc():
      return hw.StructCreateOp(elem_name_values, result_type=result_type)

  raise ValueError(f"Unable to map object '{x}' to MLIR Value")


def create_type_string(ty):
  from .dialects import hw
  ty = support.type_to_pytype(ty)
  if isinstance(ty, hw.TypeAliasType):
    return ty.name
  if isinstance(ty, hw.ArrayType):
    return f"{ty.size}x" + create_type_string(ty.element_type)
  return str(ty)
