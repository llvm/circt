from circt.dialects import comb
from circt.support import NamedValueOpView, get_value

from mlir.ir import IntegerAttr, IntegerType

from functools import reduce


def infer_result_type(operands):
  types = list(map(lambda arg: get_value(arg).type, operands))
  if not types:
    raise TypeError("result type must be specified")
  all_equal = reduce(lambda t1, t2: t1 == t2, types)
  if not all_equal:
    raise TypeError(f"expected same input port types, but received {types}")
  return types[0]


# Builder base classes for non-variadic unary and binary ops.
class UnaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["input"]

  def result_names(self):
    return ["result"]


def UnaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, result_type, **kwargs):
      return UnaryOpBuilder(cls, result_type, kwargs)

  return _Class


class ExtractOpBuilder(UnaryOpBuilder):

  def __init__(self, low_bit, data_type, input_port_mapping={}, **kwargs):
    low_bit = IntegerAttr.get(IntegerType.get_signless(32), low_bit)
    super().__init__(comb.ExtractOp, data_type, input_port_mapping, [],
                     [low_bit], **kwargs)


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


def BinaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, result_type=None, **kwargs):
      if not result_type:
        result_type = infer_result_type(kwargs.values())
      return BinaryOpBuilder(cls, result_type, kwargs)

  return _Class


# Base classes for the variadic ops.
def VariadicOp(base):

  class _Class(base):

    @classmethod
    def create(cls, *args):
      result_type = infer_result_type(args)
      return cls(result_type, args)

  return _Class


# Base class for miscellaneous ops that can't be abstracted but should provide a
# create method for uniformity.
def CreatableOp(base):

  class _Class(base):

    @classmethod
    def create(cls, *args, **kwargs):
      return cls(*args, **kwargs)

  return _Class


# Sugar classes for the various non-variadic unary ops.
class ExtractOp:

  @staticmethod
  def create(low_bit, result_type, **kwargs):
    return ExtractOpBuilder(low_bit, result_type, kwargs)


@UnaryOp
class ParityOp:
  pass


@UnaryOp
class SExtOp:
  pass


# Sugar classes for the various non-variadic binary ops.
@BinaryOp
class DivSOp:
  pass


@BinaryOp
class DivUOp:
  pass


@BinaryOp
class ModSOp:
  pass


@BinaryOp
class ModUOp:
  pass


@BinaryOp
class ShlOp:
  pass


@BinaryOp
class ShrSOp:
  pass


@BinaryOp
class ShrUOp:
  pass


@BinaryOp
class SubOp:
  pass


# Sugar classes for the variadic ops.
@VariadicOp
class AddOp:
  pass


@VariadicOp
class MulOp:
  pass


@VariadicOp
class AndOp:
  pass


@VariadicOp
class OrOp:
  pass


@VariadicOp
class XorOp:
  pass


@VariadicOp
class MergeOp:
  pass


# Sugar classes for miscellaneous ops.
class ConcatOp:

  @classmethod
  def create(cls, result_type, *args, **kwargs):
    return cls(result_type, args, **kwargs)


@CreatableOp
class MuxOp:
  pass
