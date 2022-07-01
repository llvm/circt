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
_hidden_filenames = set(["functools.py"])


def get_user_loc() -> ir.Location:
  import traceback
  stack = reversed(traceback.extract_stack())
  for frame in stack:
    fn = os.path.split(frame.filename)[1]
    if frame.filename in _local_files or fn in _hidden_filenames:
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
    if result_type is None:
      result_type = self.type
    val = _obj_to_value(obj, self.type, result_type)
    support.connect(self, val)


def _obj_to_value(x, type, result_type=None):
  """Convert a python object to a CIRCT value, given the CIRCT type."""
  if x is None:
    raise ValueError(
        "Encountered 'None' when trying to build hardware for python value.")
  from .value import Value
  from .dialects import hw
  from .pycde_types import (TypeAliasType, ArrayType, StructType, BitVectorType,
                            Type)

  if isinstance(x, Value):
    return x

  type = Type(type)
  if isinstance(type, TypeAliasType):
    return _obj_to_value(x, type.inner_type, type)

  if result_type is None:
    result_type = type
  else:
    result_type = Type(result_type)
    assert isinstance(result_type, TypeAliasType) or result_type == type

  val = support.get_value(x)
  # If x is already a valid value, just return it.
  if val is not None:
    if val.type != result_type:
      raise ValueError(f"Expected {result_type}, got {val.type}")
    return val

  if isinstance(x, int):
    if not isinstance(type, BitVectorType):
      raise ValueError(f"Int can only be converted to hw int, not '{type}'")
    with get_user_loc():
      return hw.ConstantOp(type, x)

  if isinstance(x, (list, tuple)):
    if not isinstance(type, ArrayType):
      raise ValueError(f"List is only convertable to hw array, not '{type}'")
    elemty = result_type.element_type
    if len(x) != type.size:
      raise ValueError("List must have same size as array "
                       f"{len(x)} vs {type.size}")
    list_of_vals = list(map(lambda x: _obj_to_value(x, elemty), x))
    # CIRCT's ArrayCreate op takes the array in reverse order.
    with get_user_loc():
      return hw.ArrayCreateOp(reversed(list_of_vals))

  if isinstance(x, dict):
    if not isinstance(type, StructType):
      raise ValueError(f"Dict is only convertable to hw struct, not '{type}'")
    elem_name_values = []
    for (fname, ftype) in type.fields:
      if fname not in x:
        raise ValueError(f"Could not find expected field: {fname}")
      elem_name_values.append((fname, _obj_to_value(x[fname], ftype)))
      x.pop(fname)
    if len(x) > 0:
      raise ValueError(f"Extra fields specified: {x}")
    with get_user_loc():
      return hw.StructCreateOp(elem_name_values, result_type=result_type._type)

  raise ValueError(f"Unable to map object '{x}' to MLIR Value")


def _infer_type(x):
  """Infer the CIRCT type from a python object. Only works on lists."""
  from .pycde_types import types
  from .value import Value
  if isinstance(x, Value):
    return x.type

  if isinstance(x, (list, tuple)):
    list_types = [_infer_type(i) for i in x]
    list_type = list_types[0]
    if not all([i == list_type for i in list_types]):
      raise ValueError("CIRCT array must be homogenous, unlike object")
    return types.array(list_type, len(x))
  if isinstance(x, int):
    raise ValueError(f"Cannot infer width of {x}")
  if isinstance(x, dict):
    raise ValueError(f"Cannot infer struct field order of {x}")
  return None


def _obj_to_value_infer_type(value):
  """Infer the CIRCT type, then convert the Python object to a CIRCT Value of
  that type."""
  type = _infer_type(value)
  if type is None:
    raise ValueError(f"Cannot infer CIRCT type from '{value}")
  return _obj_to_value(value, type)


def create_type_string(ty):
  from .dialects import hw
  ty = support.type_to_pytype(ty)
  if isinstance(ty, hw.TypeAliasType):
    return ty.name
  if isinstance(ty, hw.ArrayType):
    return f"{ty.size}x" + create_type_string(ty.element_type)
  return str(ty)


def attributes_of_type(o, T):
  """Filter the attributes of an object 'o' to only those of type 'T'."""
  return {a: getattr(o, a) for a in dir(o) if isinstance(getattr(o, a), T)}
