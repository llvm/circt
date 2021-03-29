from mlir.ir import *

class WireOp:
  """Specialization for the wire op class."""

  def __init__(self, name, resultType, *, loc=None, ip=None):
    """
    Create a wire with the given name and type. Needed because WireOp
    skips the default builders, so an __init__ is not generated.
    """
    from circt.dialects.rtl import InOutType
    operands = []
    results = [InOutType.get(resultType)]
    attributes = {}
    attributes['name'] = StringAttr.get(str(name))
    super().__init__(self.build_generic(
        attributes=attributes, results=results, operands=operands,
        loc=loc, ip=ip))

