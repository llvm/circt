from mlir.ir import OpView, StringAttr


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
