from circt.dialects import hwarith
from circt.support import NamedValueOpView, get_value

from mlir.ir import IntegerAttr, IntegerType, Type


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


def BinaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, lhs=None, rhs=None, result_type=None):
      return cls([get_value(lhs), get_value(rhs)])

  return _Class


@BinaryOp
class DivOp:
  pass


@BinaryOp
class SubOp:
  pass


@BinaryOp
class AddOp:
  pass


@BinaryOp
class MulOp:
  pass
