#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Type

from . import hw, msft as _msft
from . import _hw_ops_ext as _hw_ext
from .. import support

from .. import ir as _ir


class InstanceBuilder(support.NamedValueOpView):
  """Helper class to incrementally construct an instance of a module."""

  def __init__(self,
               module,
               name,
               input_port_mapping,
               *,
               parameters=None,
               target_design_partition=None,
               loc=None,
               ip=None):
    self.module = module
    instance_name = hw.InnerSymAttr.get(_ir.StringAttr.get(name))
    module_name = _ir.FlatSymbolRefAttr.get(_ir.StringAttr(module.name).value)
    pre_args = [instance_name, module_name]
    if parameters is not None:
      parameters = _hw_ext.create_parameters(parameters, module)
    else:
      parameters = []
    post_args = []
    results = module.type.results

    super().__init__(
        _msft.InstanceOp,
        results,
        input_port_mapping,
        pre_args,
        post_args,
        parameters=_ir.ArrayAttr.get(parameters),
        targetDesignPartition=target_design_partition,
        loc=loc,
        ip=ip,
    )

  def create_default_value(self, index, data_type, arg_name):
    type = self.module.type.inputs[index]
    return support.BackedgeBuilder.create(type,
                                          arg_name,
                                          self,
                                          instance_of=self.module)

  def operand_names(self):
    arg_names = _ir.ArrayAttr(self.module.attributes["argNames"])
    arg_name_attrs = map(_ir.StringAttr, arg_names)
    return list(map(lambda s: s.value, arg_name_attrs))

  def result_names(self):
    arg_names = _ir.ArrayAttr(self.module.attributes["resultNames"])
    arg_name_attrs = map(_ir.StringAttr, arg_names)
    return list(map(lambda s: s.value, arg_name_attrs))


class MSFTModuleLike:
  """Custom Python base class for module-like operations."""

  def __init__(
      self,
      name,
      input_ports=[],
      output_ports=[],
      *,
      parameters=[],
      attributes={},
      body_builder=None,
      loc=None,
      ip=None,
  ):
    """
    Create a module-like with the provided `name`, `input_ports`, and
    `output_ports`.
    - `name` is a string representing the module name.
    - `input_ports` is a list of pairs of string names and mlir.ir types.
    - `output_ports` is a list of pairs of string names and mlir.ir types.
    - `body_builder` is an optional callback, when provided a new entry block
      is created and the callback is invoked with the new op as argument within
      an InsertionPoint context already set for the block. The callback is
      expected to insert a terminator in the block.
    """
    # Copy the mutable default arguments. 'Cause python.
    input_ports = list(input_ports)
    output_ports = list(output_ports)
    parameters = list(parameters)
    attributes = dict(attributes)
    operands = []
    results = []
    attributes["sym_name"] = _ir.StringAttr.get(str(name))
    input_types = []
    input_names = []
    input_locs = []
    unknownLoc = _ir.Location.unknown().attr
    for (i, (port_name, port_type)) in enumerate(input_ports):
      input_types.append(port_type)
      input_names.append(_ir.StringAttr.get(str(port_name)))
      input_locs.append(unknownLoc)
    attributes["argNames"] = _ir.ArrayAttr.get(input_names)
    attributes["argLocs"] = _ir.ArrayAttr.get(input_locs)
    output_types = []
    output_names = []
    output_locs = []
    for (i, (port_name, port_type)) in enumerate(output_ports):
      output_types.append(port_type)
      output_names.append(StringAttr.get(str(port_name)))
      output_locs.append(unknownLoc)
    attributes["resultNames"] = _ir.ArrayAttr.get(output_names)
    attributes["resultLocs"] = _ir.ArrayAttr.get(output_locs)
    if len(parameters) > 0 or "parameters" not in attributes:
      attributes["parameters"] = _ir.ArrayAttr.get(parameters)

    attributes["function_type"] = _ir.TypeAttr.get(
        _ir.FunctionType.get(inputs=input_types, results=output_types))

    super().__init__(
        self.build_generic(attributes=attributes,
                           results=results,
                           operands=operands,
                           loc=loc,
                           ip=ip))

    if body_builder:
      entry_block = self.add_entry_block()

      with InsertionPoint(entry_block):
        with support.BackedgeBuilder():
          outputs = body_builder(self)
          _create_output_op(name, output_ports, entry_block, outputs)

  @property
  def type(self):
    return _ir.FunctionType(
        _ir.TypeAttr(self.attributes["function_type"]).value)

  @property
  def name(self):
    return self.attributes["sym_name"]

  @property
  def is_external(self):
    return len(self.regions[0].blocks) == 0

  @property
  def parameters(self) -> List[hw.ParamDeclAttr]:
    return [
        hw.ParamDeclAttr(a) for a in ArrayAttr(self.attributes["parameters"])
    ]

  def instantiate(self,
                  name: str,
                  parameters: Dict[str, object] = {},
                  results=None,
                  loc=None,
                  ip=None,
                  **kwargs):
    return InstanceBuilder(self,
                           name,
                           kwargs,
                           parameters=parameters,
                           results=results,
                           loc=loc,
                           ip=ip)


def _create_output_op(cls_name, output_ports, entry_block, bb_ret):
  """Create the hw.OutputOp from the body_builder return."""
  # Determine if the body already has an output op.
  block_len = len(entry_block.operations)
  if block_len > 0:
    last_op = entry_block.operations[block_len - 1]
    if isinstance(last_op, hw.OutputOp):
      # If it does, the return from body_builder must be None.
      if bb_ret is not None and bb_ret != last_op:
        raise support.ConnectionError(
            f"In {cls_name}, cannot return value from body_builder and "
            "create hw.OutputOp")
      return
  # If builder didn't create an output op and didn't return anything, this op
  # mustn't have any outputs.
  if bb_ret is None:
    if len(output_ports) == 0:
      hw.OutputOp([])
      return
    raise support.ConnectionError(
        f"In {cls_name}, must return module output values")
  # Now create the output op depending on the object type returned
  outputs: list[Value] = list()
  # Only acceptable return is a dict of port, value mappings.
  if not isinstance(bb_ret, dict):
    raise support.ConnectionError(
        f"In {cls_name}, can only return a dict of port, value mappings "
        "from body_builder.")
  # A dict of `OutputPortName` -> ValueLike must be converted to a list in port
  # order.
  unconnected_ports = []
  for (name, port_type) in output_ports:
    if name not in bb_ret:
      unconnected_ports.append(name)
      outputs.append(None)
    else:
      val = support.get_value(bb_ret[name])
      if val is None:
        field_type = type(bb_ret[name])
        raise TypeError(
            f"In {cls_name}, body_builder return doesn't support type "
            f"'{field_type}'")
      if val.type != port_type:
        if isinstance(port_type, hw.TypeAliasType) and \
           port_type.inner_type == val.type:
          val = hw.BitcastOp.create(port_type, val).result
        else:
          raise TypeError(
              f"In {cls_name}, output port '{name}' type ({val.type}) doesn't "
              f"match declared type ({port_type})")
      outputs.append(val)
      bb_ret.pop(name)
  if len(unconnected_ports) > 0:
    raise support.UnconnectedSignalError(cls_name, unconnected_ports)
  if len(bb_ret) > 0:
    raise support.ConnectionError(
        f"Could not map the following to output ports in {cls_name}: " +
        ",".join(bb_ret.keys()))
  hw.OutputOp(outputs)


class MSFTModuleOp(MSFTModuleLike):

  def __init__(
      self,
      name,
      input_ports=[],
      output_ports=[],
      parameters: _ir.DictAttr = None,
      file_name: str = None,
      attributes=None,
      loc=None,
      ip=None,
  ):
    if attributes is None:
      attributes = {}
    if parameters is not None:
      attributes["parameters"] = parameters
    else:
      attributes["parameters"] = _ir.DictAttr.get({})
    if file_name is not None:
      attributes["fileName"] = _ir.StringAttr.get(file_name)
    super().__init__(name,
                     input_ports,
                     output_ports,
                     attributes=attributes,
                     loc=loc,
                     ip=ip)

  def instantiate(self, name: str, loc=None, ip=None, **kwargs):
    return InstanceBuilder(self, name, kwargs, loc=loc, ip=ip)

  def add_entry_block(self):
    self.body.blocks.append(*self.type.inputs)
    return self.body.blocks[0]

  @property
  def body(self):
    return self.regions[0]

  @property
  def entry_block(self):
    return self.regions[0].blocks[0]

  @property
  def parameters(self):
    return [
        hw.ParamDeclAttr.get(p.name, p.attr.type, p.attr)
        for p in _ir.DictAttr(self.attributes["parameters"])
    ]

  @property
  def childAppIDBases(self):
    if "childAppIDBases" not in self.attributes:
      return None
    bases = self.attributes["childAppIDBases"]
    if bases is None:
      return None
    return [_ir.StringAttr(n) for n in _ir.ArrayAttr(bases)]


class MSFTModuleExternOp(MSFTModuleLike):

  def instantiate(self,
                  name: str,
                  parameters=None,
                  results=None,
                  loc=None,
                  ip=None,
                  **kwargs):
    return InstanceBuilder(self,
                           name,
                           kwargs,
                           parameters=parameters,
                           loc=loc,
                           ip=ip)


class PhysicalRegionOp:

  def add_bounds(self, bounds):
    existing_bounds = [b for b in _ir.ArrayAttr(self.attributes["bounds"])]
    existing_bounds.append(bounds)
    new_bounds = _ir.ArrayAttr.get(existing_bounds)
    self.attributes["bounds"] = new_bounds


class InstanceOp:

  @property
  def moduleName(self):
    return _ir.FlatSymbolRefAttr(self.attributes["moduleName"])


class EntityExternOp:

  @staticmethod
  def create(symbol, metadata=""):
    symbol_attr = support.var_to_attribute(symbol)
    metadata_attr = support.var_to_attribute(metadata)
    return _msft.EntityExternOp(symbol_attr, metadata_attr)


class InstanceHierarchyOp:

  @staticmethod
  def create(root_mod, instance_name=None):
    hier = _msft.InstanceHierarchyOp(root_mod, instName=instance_name)
    hier.body.blocks.append()
    return hier

  @property
  def top_module_ref(self):
    return self.attributes["topModuleRef"]


class DynamicInstanceOp:

  @staticmethod
  def create(name_ref):
    inst = _msft.DynamicInstanceOp(name_ref)
    inst.body.blocks.append()
    return inst

  @property
  def instance_path(self):
    path = []
    next = self
    while isinstance(next, DynamicInstanceOp):
      path.append(next.attributes["instanceRef"])
      next = next.operation.parent.opview
    path.reverse()
    return _ir.ArrayAttr.get(path)

  @property
  def instanceRef(self):
    return self.attributes["instanceRef"]


class PDPhysLocationOp:

  @property
  def loc(self):
    return _msft.PhysLocationAttr(self.attributes["loc"])
