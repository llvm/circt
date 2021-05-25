from circt.support import BackedgeBuilder, NamedValueOpView

from mlir.ir import IntegerType, OpView, StringAttr, Value


class CompRegBuilder(NamedValueOpView):
  INPUT_PORT_NAMES = ["input", "clk"]

  def __init__(self,
               data_type,
               input_port_mapping={},
               *,
               reset=None,
               reset_value=None,
               name=None,
               loc=None,
               ip=None):
    # Lazily import dependencies to avoid cyclic dependencies.
    from ._seq_ops_gen import CompRegOp

    backedges = {}
    operand_indices = {}
    operand_values = []
    result_indices = {"data": 0}
    for i in range(len(self.INPUT_PORT_NAMES)):
      arg_name = self.INPUT_PORT_NAMES[i]
      operand_indices[arg_name] = i
      if arg_name in input_port_mapping:
        value = input_port_mapping[arg_name]
        if not isinstance(value, Value):
          value = input_port_mapping[arg_name].value
        operand = value
      else:
        i1 = IntegerType.get_signless(1)
        operand_type = data_type if arg_name == "input" else i1
        backedge = BackedgeBuilder.create(operand_type, arg_name, self)
        backedges[i] = backedge
        operand = backedge.result
      operand_values.append(operand)

    compreg = CompRegOp(data_type,
                        operand_values[0],
                        operand_values[1],
                        reset=reset,
                        reset_value=reset_value,
                        name=name,
                        loc=loc,
                        ip=ip)

    super().__init__(compreg, operand_indices, result_indices, backedges)


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
    if name is not None:
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

  @staticmethod
  def create(*args, **kwargs):
    return CompRegBuilder(*args, **kwargs)
