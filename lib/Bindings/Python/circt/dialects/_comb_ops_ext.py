from circt.dialects import comb
from circt.support import NamedValueOpView

from mlir.ir import IntegerAttr, IntegerType


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
    def create(cls, result_type, **kwargs):
      return BinaryOpBuilder(cls, result_type, kwargs)

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
