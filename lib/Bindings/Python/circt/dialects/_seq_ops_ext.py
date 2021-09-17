from circt.support import BackedgeBuilder, NamedValueOpView

from mlir.ir import IntegerType, OpView, StringAttr


class CompRegBuilder(NamedValueOpView):

  def operand_names(self):
    return ["input", "clk"]

  def result_names(self):
    return ["data"]

  def create_initial_value(self, index, data_type, arg_name):
    if arg_name == "input":
      operand_type = data_type
    else:
      operand_type = IntegerType.get_signless(1)
    return BackedgeBuilder.create(operand_type, arg_name, self)


class CompRegOp:

  def __init__(self,
               data_type,
               input,
               clk,
               *,
               reset=None,
               reset_value=None,
               name=None,
               loc=None,
               ip=None):
    operands = []
    results = []
    attributes = {}
    results.append(data_type)
    operands.append(input)
    operands.append(clk)
    if reset is not None:
      operands.append(reset)
    if reset_value is not None:
      operands.append(reset_value)
    if name is None:
      attributes["name"] = StringAttr.get("")
    else:
      attributes["name"] = StringAttr.get(name)

    OpView.__init__(
        self,
        self.build_generic(
            attributes=attributes,
            results=results,
            operands=operands,
            loc=loc,
            ip=ip,
        ),
    )

  @classmethod
  def create(cls,
             result_type,
             reset=None,
             reset_value=None,
             name=None,
             **kwargs):
    return CompRegBuilder(cls,
                          result_type,
                          kwargs,
                          reset=reset,
                          reset_value=reset_value,
                          name=name)
