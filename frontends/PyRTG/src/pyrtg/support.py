#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import support, ir
from .core import Value, Type

from typing import Union


def _FromCirctValue(value: ir.Value) -> Value:
  return _FromCirctType(value.type)._wrap(value)


def _FromCirctType(type: Union[ir.Type, Type]) -> Type:
  if isinstance(type, Type):
    return type

  type = support.type_to_pytype(type)

  from .rtg import rtg
  from .rtgtest import rtgtest
  if isinstance(type, rtg.ArrayType):
    from .arrays import ArrayType
    return ArrayType(_FromCirctType(type.element_type))
  if isinstance(type, rtg.BagType):
    from .bags import BagType
    return BagType(_FromCirctType(type.element_type))
  if isinstance(type, rtg.SetType):
    from .sets import SetType
    return SetType(_FromCirctType(type.element_type))
  if isinstance(type, ir.IntegerType) and type.is_signless:
    if type.width == 1:
      from .booleans import BoolType
      return BoolType()
    else:
      from .immediates import ImmediateType
      return ImmediateType(type.width)
  if isinstance(type, ir.IndexType):
    from .integers import IntegerType
    return IntegerType()
  if isinstance(type, rtg.LabelType):
    from .labels import LabelType
    return LabelType()
  if isinstance(type, rtg.StringType):
    from .strings import StringType
    return StringType()
  if isinstance(type, rtg.SequenceType):
    from .sequences import SequenceType
    return SequenceType(
        [_FromCirctType(type.get_element(i)) for i in range(type.num_elements)])
  if isinstance(type, rtg.RandomizedSequenceType):
    from .sequences import RandomizedSequenceType
    return RandomizedSequenceType()
  if isinstance(type, rtgtest.IntegerRegisterType):
    from .resources import IntegerRegisterType
    return IntegerRegisterType()
  if isinstance(type, rtgtest.CPUType):
    from .contexts import CPUCoreType
    return CPUCoreType()
  if isinstance(type, rtg.TupleType):
    from .tuples import TupleType
    return TupleType([_FromCirctType(ty) for ty in type.fields])
  if isinstance(type, rtg.MemoryType):
    from .memories import MemoryType
    return MemoryType(type.address_width)
  if isinstance(type, rtg.MemoryBlockType):
    from .memories import MemoryBlockType
    return MemoryBlockType(type.address_width)
  if isinstance(type, rtg.ContinuationType):
    from .effects import ContinuationType
    resume = type.resume_type
    if isinstance(resume, ir.NoneType):
      from .effects import VoidType
      return ContinuationType(VoidType())
    return ContinuationType(_FromCirctType(resume))
  raise ValueError("unsupported type")


def _collect_values_recursively(obj, path, args, arg_names, visited):
  if obj is None or id(obj) in visited:
    return args, arg_names

  visited.add(id(obj))

  # Base case
  if isinstance(obj, Value):
    args.append(obj)
    arg_names.append(path)
    return args, arg_names

  # Recursive case
  try:
    for attr_name, attr_value in obj.__dict__.items():
      _collect_values_recursively(attr_value, f"{path}.{attr_name}", args,
                                  arg_names, visited)
  except AttributeError:
    pass

  return args, arg_names


def _to_circt(arg):
  """Convert frontend operands for an explicit dialect operation call."""
  from .sequences import SequenceDeclaration
  if isinstance(arg, (Value, SequenceDeclaration)):
    return arg._get_ssa_value()
  if isinstance(arg, Type):
    return arg._codegen()
  if isinstance(arg, (list, tuple)):
    return [_to_circt(a) for a in arg]
  return arg


def _create(op_class, *args, **kwargs):
  """Create a dialect operation without changing the binding's return type."""
  args = [_to_circt(arg) for arg in args]
  kwargs = {k: _to_circt(v) for k, v in kwargs.items()}
  return op_class(*args, **kwargs)
