#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .circt import support, ir
from .core import Value


def _FromCirctValue(value: ir.Value) -> Value:
  type = support.type_to_pytype(value.type)
  from .rtg import rtg
  if isinstance(type, rtg.LabelType):
    from .labels import Label
    return Label(value)
  if isinstance(type, rtg.SetType):
    from .sets import Set
    return Set(value)
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
            if isinstance(arg, Value):
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
