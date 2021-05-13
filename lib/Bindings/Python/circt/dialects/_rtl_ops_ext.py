from typing import Dict, Optional, Sequence

import inspect

from circt.support import BackedgeBuilder

from mlir.ir import *


class InstanceBuilder:
  """Helper class to incrementally construct an instance of a module."""

  def __init__(self,
               parent_module,
               module,
               name,
               input_port_mapping,
               *,
               parameters={},
               sym_name=None,
               loc=None,
               ip=None):
    # Lazily import dependencies to avoid cyclic dependencies.
    from ._rtl_ops_gen import InstanceOp

    # Create mappings from port name to value, index, and potentially backedge.
    self.parent_module = parent_module
    self.mod = module
    self.operand_indices = {}
    self.operand_values = []
    self.result_indices = {}
    self.backedges = {}

    arg_names = ArrayAttr(module.attributes["argNames"])
    for i in range(len(arg_names)):
      arg_name = StringAttr(arg_names[i]).value
      self.operand_indices[arg_name] = i

      if arg_name in input_port_mapping:
        self.operand_values.append(input_port_mapping[arg_name])
      else:
        type = module.type.inputs[i]
        backedge = self.parent_module.create_backedge(type, self)
        self.backedges[i] = backedge
        self.operand_values.append(backedge)

    result_names = ArrayAttr(module.attributes["resultNames"])
    for i in range(len(result_names)):
      result_name = StringAttr(result_names[i]).value
      self.result_indices[result_name] = i

    # Actually build the InstanceOp.
    instance_name = StringAttr.get(name)
    module_name = FlatSymbolRefAttr.get(StringAttr(self.mod.name).value)
    parameters = {k: Attribute.parse(str(v)) for (k, v) in parameters.items()}
    parameters = DictAttr.get(parameters)
    if sym_name:
      sym_name = StringAttr.get(sym_name)
    self.instance = InstanceOp(
        self.mod.type.results,
        instance_name,
        module_name,
        self.operand_values,
        parameters,
        sym_name,
        loc=loc,
        ip=ip,
    )

  def __getattr__(self, name):
    # Check for the attribute in the arg name set.
    if name in self.operand_indices:
      index = self.operand_indices[name]
      return self.instance.inputs[index]

    # Check for the attribute in the result name set.
    if name in self.result_indices:
      index = self.result_indices[name]
      return self.instance.results[index]

    # If we fell through to here, the name isn't a result.
    raise AttributeError(f"unknown port name {name}")

  def set_input_port(self, name, value):
    # Check for the attribute in the arg name set.
    if name in self.operand_indices:
      # Put the value into the instance.
      index = self.operand_indices[name]
      self.instance.inputs[index] = value
      self.parent_module.remove_backedge(self.backedges[index])
      return

    # If we fell through to here, the name isn't an arg.
    raise AttributeError(f"unknown input port name {name}")

  @property
  def operation(self):
    """Get the operation associated with this builder."""
    return self.instance.operation

  @property
  def module(self):
    """Get the module associated with this builder."""
    return self.mod.operation


class ModuleLike:
  """Custom Python base class for module-like operations."""

  def __init__(
      self,
      name,
      input_ports=[],
      output_ports=[],
      *,
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
    operands = []
    results = []
    attributes = {}
    attributes["sym_name"] = StringAttr.get(str(name))

    input_types = []
    input_names = []
    self.input_indices = {}
    input_ports = list(input_ports)
    for i in range(len(input_ports)):
      port_name, port_type = input_ports[i]
      input_types.append(port_type)
      input_names.append(StringAttr.get(str(port_name)))
      self.input_indices[port_name] = i
    attributes["argNames"] = ArrayAttr.get(input_names)

    output_types = []
    output_names = []
    output_ports = list(output_ports)
    self.output_indices = {}
    for i in range(len(output_ports)):
      port_name, port_type = output_ports[i]
      output_types.append(port_type)
      output_names.append(StringAttr.get(str(port_name)))
      self.output_indices[port_name] = i
    attributes["resultNames"] = ArrayAttr.get(output_names)

    attributes["type"] = TypeAttr.get(
        FunctionType.get(inputs=input_types, results=output_types))

    super().__init__(
        self.build_generic(attributes=attributes,
                           results=results,
                           operands=operands,
                           loc=loc,
                           ip=ip))

    if body_builder:
      entry_block = self.add_entry_block()
      with InsertionPoint(entry_block):
        with BackedgeBuilder() as bb:
          self.backedge_builder = bb
          body_builder(self)

  @property
  def type(self):
    return FunctionType(TypeAttr(self.attributes["type"]).value)

  @property
  def name(self):
    return self.attributes["sym_name"]

  @property
  def is_external(self):
    return len(self.regions[0].blocks) == 0

  def create(self,
             module,
             name: str,
             input_port_mapping: Dict[str, Value] = {},
             parameters: Dict[str, object] = {},
             loc=None,
             ip=None):
    return InstanceBuilder(module,
                           self,
                           name,
                           input_port_mapping,
                           parameters=parameters,
                           loc=loc,
                           ip=ip)

  def create_backedge(self, type, builder):
    assert self.backedge_builder, "No backedge builder initialized."
    return self.backedge_builder.create(type, builder)

  def remove_backedge(self, backedge):
    assert self.backedge_builder, "No backedge builder initialized."
    self.backedge_builder.remove(backedge)


class RTLModuleOp(ModuleLike):
  """Specialization for the RTL module op class."""

  @property
  def body(self):
    return self.regions[0]

  @property
  def entry_block(self):
    return self.regions[0].blocks[0]

  # Support attribute access to block arguments by name
  def __getattr__(self, name):
    if name in self.input_indices:
      index = self.input_indices[name]
      return self.entry_block.arguments[index]
    raise AttributeError(f"unknown input port name {name}")

  def add_entry_block(self):
    if not self.is_external:
      raise IndexError('The module already has an entry block')
    self.body.blocks.append(*self.type.inputs)
    return self.body.blocks[0]

  @classmethod
  def from_py_func(RTLModuleOp,
                   *inputs: Type,
                   results: Optional[Sequence[Type]] = None,
                   name: Optional[str] = None):
    """Decorator to define an MLIR RTLModuleOp specified as a python function.
    Requires that an `mlir.ir.InsertionPoint` and `mlir.ir.Location` are
    active for the current thread (i.e. established in a `with` block).
    When applied as a decorator to a Python function, an entry block will
    be constructed for the RTLModuleOp with types as specified in `*inputs`. The
    block arguments will be passed positionally to the Python function. In
    addition, if the Python function accepts keyword arguments generally or
    has a corresponding keyword argument, the following will be passed:
      * `module_op`: The `module` op being defined.
    By default, the function name will be the Python function `__name__`. This
    can be overriden by passing the `name` argument to the decorator.
    If `results` is not specified, then the decorator will implicitly
    insert a `OutputOp` with the `Value`'s returned from the decorated
    function. It will also set the `RTLModuleOp` type with the actual return
    value types. If `results` is specified, then the decorated function
    must return `None` and no implicit `OutputOp` is added (nor are the result
    types updated). The implicit behavior is intended for simple, single-block
    cases, and users should specify result types explicitly for any complicated
    cases.
    The decorated function can further be called from Python and will insert
    a `InstanceOp` at the then-current insertion point, returning either None (
    if no return values), a unary Value (for one result), or a list of Values).
    This mechanism cannot be used to emit recursive calls (by construction).
    """

    def decorator(f):
      from circt.dialects import rtl
      # Introspect the callable for optional features.
      sig = inspect.signature(f)
      has_arg_module_op = False
      for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
          has_arg_module_op = True
        if param.name == "module_op" and (param.kind
                                          == param.POSITIONAL_OR_KEYWORD or
                                          param.kind == param.KEYWORD_ONLY):
          has_arg_module_op = True

      # Emit the RTLModuleOp.
      implicit_return = results is None
      symbol_name = name or f.__name__
      input_names = [v.name for v in sig.parameters.values()]
      input_types = [port_type for port_type in inputs]
      input_ports = zip(input_names, input_types)

      if implicit_return:
        output_ports = []
      else:
        result_types = [port_type for port_type in results]
        output_ports = zip([None] * len(result_types), result_types)

      module_op = RTLModuleOp(name=symbol_name,
                              input_ports=input_ports,
                              output_ports=output_ports)
      with InsertionPoint(module_op.add_entry_block()):
        module_args = module_op.entry_block.arguments
        module_kwargs = {}
        if has_arg_module_op:
          module_kwargs["module_op"] = module_op
        return_values = f(*module_args, **module_kwargs)
        if not implicit_return:
          return_types = list(results)
          assert return_values is None, (
              "Capturing a python function with explicit `results=` "
              "requires that the wrapped function returns None.")
        else:
          # Coerce return values, add OutputOp and rewrite func type.
          if return_values is None:
            return_values = []
          elif isinstance(return_values, Value):
            return_values = [return_values]
          else:
            return_values = list(return_values)
          rtl.OutputOp(return_values)
          # Recompute the function type.
          return_types = [v.type for v in return_values]
          function_type = FunctionType.get(inputs=inputs, results=return_types)
          module_op.attributes["type"] = TypeAttr.get(function_type)
          # Set required resultNames attribute. Could we infer real names here?
          resultNames = [
              StringAttr.get('result' + str(i))
              for i in range(len(return_values))
          ]
          module_op.attributes["resultNames"] = ArrayAttr.get(resultNames)

      def emit_instance_op(*call_args):
        call_op = rtl.InstanceOp(return_types, StringAttr.get(''),
                                 FlatSymbolRefAttr.get(symbol_name), call_args,
                                 DictAttr.get({}), None)
        if return_types is None:
          return None
        elif len(return_types) == 1:
          return call_op.result
        else:
          return call_op.results

      wrapped = emit_instance_op
      wrapped.__name__ = f.__name__
      wrapped.module_op = module_op
      return wrapped

    return decorator


class RTLModuleExternOp(ModuleLike):
  """Specialization for the RTL module op class."""
  pass
