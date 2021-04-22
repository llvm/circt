from mlir.ir import OpView, StringAttr


class CompRegOp:

  def __init__(self,
               data,
               input,
               clk,
               reset,
               resetValue,
               *,
               name=None,
               loc=None,
               ip=None):
    operands = []
    results = []
    attributes = {}
    results.append(data)
    operands.append(input)
    operands.append(clk)
    if reset is not None:
      operands.append(reset)
    if resetValue is not None:
      operands.append(resetValue)
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
