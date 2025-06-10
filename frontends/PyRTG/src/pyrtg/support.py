#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import support, ir
from .core import Value, Type

from typing import Union


def _FromCirctValue(value: ir.Value) -> Value:
  type = support.type_to_pytype(value.type)
  from .rtg import rtg
  from .rtgtest import rtgtest
  if isinstance(type, rtg.ArrayType):
    from .arrays import Array
    return Array(value)
  if isinstance(type, rtg.LabelType):
    from .labels import Label
    return Label(value)
  if isinstance(type, rtg.SetType):
    from .sets import Set
    return Set(value)
  if isinstance(type, rtg.BagType):
    from .bags import Bag
    return Bag(value)
  if isinstance(type, rtg.SequenceType):
    from .sequences import Sequence
    return Sequence(value)
  if isinstance(type, rtg.RandomizedSequenceType):
    from .sequences import RandomizedSequence
    return RandomizedSequence(value)
  if isinstance(type, ir.IndexType):
    from .integers import Integer
    return Integer(value)
  if isinstance(type, ir.IntegerType) and type.width == 1:
    from .integers import Bool
    return Bool(value)
  if isinstance(type, rtgtest.IntegerRegisterType):
    from .resources import IntegerRegister
    return IntegerRegister(value)
  if isinstance(type, rtgtest.CPUType):
    from .contexts import CPUCore
    return CPUCore(value)
  if isinstance(type, rtg.ImmediateType):
    from .immediates import Immediate
    return Immediate(type.width, value)
  if isinstance(type, ir.TupleType):
    from .tuples import Tuple
    return Tuple(value)
  if isinstance(type, rtg.MemoryType):
    from .memories import Memory
    return Memory(value)
  if isinstance(type, rtg.MemoryBlockType):
    from .memories import MemoryBlock
    return MemoryBlock(value)
  assert False, "Unsupported value"


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
  if isinstance(type, rtg.ImmediateType):
    from .immediates import ImmediateType
    return ImmediateType(type.width)
  if isinstance(type, ir.IntegerType) and type.is_signless and type.width == 1:
    from .integers import BoolType
    return BoolType()
  if isinstance(type, ir.IndexType):
    from .integers import IntegerType
    return IntegerType()
  if isinstance(type, rtg.LabelType):
    from .labels import LabelType
    return LabelType()
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
  if isinstance(type, ir.TupleType):
    from .tuples import TupleType
    return TupleType(
        [_FromCirctType(type.get_type(i)) for i in range(type.num_types)])
  if isinstance(type, rtg.MemoryType):
    from .memories import MemoryType
    return MemoryType(type.address_width)
  if isinstance(type, rtg.MemoryBlockType):
    from .memories import MemoryBlockType
    return MemoryBlockType(type.address_width)
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


def wrap_opviews_with_values(dialect, module_name, excluded=[]):
  """
  Wraps all of a dialect's OpView classes to have their create method return a
  Value instead of an OpView.
  """

  import sys
  module = sys.modules[module_name]

  for attr in dir(dialect):
    cls = getattr(dialect, attr)

    if attr not in excluded and isinstance(cls, type) and issubclass(
        cls, ir.OpView):

      def specialize_create(cls):

        def create(*args, **kwargs):
          # If any of the arguments are 'pyrtg.Value', we need to convert them.
          def to_circt(arg):
            from .sequences import SequenceDeclaration
            if isinstance(arg, (Value, SequenceDeclaration)):
              return arg._get_ssa_value()
            if isinstance(arg, Type):
              return arg._codegen()
            if isinstance(arg, (list, tuple)):
              return [to_circt(a) for a in arg]
            return arg

          args = [to_circt(arg) for arg in args]
          kwargs = {k: to_circt(v) for k, v in kwargs.items()}
          # Create the OpView.
          if hasattr(cls, "create"):
            created = cls.create(*args, **kwargs)
          else:
            created = cls(*args, **kwargs)
          if isinstance(created, support.NamedValueOpView):
            created = created.opview

          # Return the wrapped values, if any.
          converted_results = tuple(
              _FromCirctValue(res) for res in created.results)
          return converted_results[0] if len(
              converted_results) == 1 else created

        return create

      wrapped_class = specialize_create(cls)
      setattr(module, attr, wrapped_class)
    else:
      setattr(module, attr, cls)
