from mlir.ir import *

class RTLModuleOp:
  """Specialization for the RTL module op class."""

  def __init__(self,
               name,
               input_ports,
               output_ports,
               *,
               body_builder=None,
               loc=None,
               ip=None):
    """
    Create a RTLModuleOp with the provided `name`, `input_ports`, and
    `output_ports`.
    - `name` is a string representing the function name.
    - `input_ports` is a list of pairs of string names and mlir.ir types.
    - `output_ports` is a list of pairs of string names and mlir.ir types.
    - `body_builder` is an optional callback, when provided a new entry block
      is created and the callback is invoked with the new op as argument within
      an InsertionPoint context already set for the block. The callback is
      expected to insert a terminator in the block.
    """
    operands = []
    results = []
    attributes = {}
    attributes['sym_name'] = StringAttr.get(str(name))

    input_types = []
    self._build_ports('arg', attributes, input_ports, input_types)

    output_types = []
    self._build_ports('result', attributes, output_ports, output_types)

    attributes["type"] = TypeAttr.get(FunctionType.get(
      inputs=input_types, results=output_types))

    super().__init__(self.build_generic(
        attributes=attributes, results=results, operands=operands,
        loc=loc, ip=ip))

    if body_builder:
      entry_block = self.add_entry_block()
      with InsertionPoint(entry_block):
        body_builder(self)

  @property
  def body(self):
    return self.regions[0]

  @property
  def type(self):
    return FunctionType(TypeAttr(self.attributes["type"]).value)

  @property
  def name(self):
    return self.attributes["sym_name"]

  @property
  def entry_block(self):
    return self.regions[0].blocks[0]

  def add_entry_block(self):
    self.body.blocks.append(*self.type.inputs)
    return self.body.blocks[0]

  def _build_ports(self, prefix, attributes, port_list, port_types):
    port_idx = 0
    for (port_name, port_type) in port_list:
      port_types.append(port_type)
      port_attrs = {}
      port_attrs['rtl.name'] = StringAttr.get(str(port_name))
      attributes[prefix + str(port_idx)] = DictAttr.get(port_attrs)
      port_idx += 1
    
