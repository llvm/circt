from circt.support import NamedValueOpView

from mlir.ir import IntegerAttr, IntegerType


# Builder base classes for non-variadic unary and binary ops.
class UnaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["input"]

  def result_names(self):
    return ["result"]


class ExtractOpBuilder(UnaryOpBuilder):

  def __init__(self, low_bit, data_type, input_port_mapping={}, **kwargs):
    low_bit = IntegerAttr.get(IntegerType.get_signless(32), low_bit)
    import circt
    super().__init__(circt.dialects.comb.ExtractOp, data_type,
                     input_port_mapping, [], [low_bit], **kwargs)


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


# Sugar classes for the various non-variadic unary ops.
class ExtractOp:

  @staticmethod
  def create(low_bit, *args, **kwargs):
    return ExtractOpBuilder(low_bit, *args, **kwargs)


class ParityOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return UnaryOpBuilder(cls, *args, **kwargs)


class SExtOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return UnaryOpBuilder(cls, *args, **kwargs)


# Sugar classes for the various non-variadic binary ops.
class DivSOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)


class DivUOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)


class ModSOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)


class ModUOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)


class ShlOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)


class ShrSOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)


class ShrUOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)


class SubOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuilder(cls, *args, **kwargs)
