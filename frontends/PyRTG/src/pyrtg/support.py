#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .circt import support, ir
from .core import Value


def _FromCirctValue(value: ir.Value) -> Value:
  type = support.type_to_pytype(value.type)
  from .rtg import rtg
  from .rtgtest import rtgtest
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
  if isinstance(type, rtgtest.IntegerRegisterType):
    from .resources import IntegerRegister
    return IntegerRegister(value)
  if isinstance(type, rtgtest.Imm5Type):
    from .resources import Imm5
    return Imm5(value)
  if isinstance(type, rtgtest.Imm12Type):
    from .resources import Imm12
    return Imm12(value)
  if isinstance(type, rtgtest.Imm13Type):
    from .resources import Imm13
    return Imm13(value)
  if isinstance(type, rtgtest.Imm21Type):
    from .resources import Imm21
    return Imm21(value)
  if isinstance(type, rtgtest.Imm32Type):
    from .resources import Imm32
    return Imm32(value)
  assert False, "Unsupported value"


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
