from circt.support import NamedValueOpView


class BinaryOpBuider(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


class DivSOp:

  @classmethod
  def create(cls, *args, **kwargs):
    return BinaryOpBuider(cls, *args, **kwargs)
