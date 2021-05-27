from circt.support import NamedValueOpView


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


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
