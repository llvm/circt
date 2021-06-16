#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import Value

from circt import support
from circt.dialects import hw
from circt.support import BackedgeBuilder, OpOperand
import circt

import mlir.ir

import inspect

OPERATION_NAMESPACE = "pycde."


class ModuleDecl:
  """Represents an input or output port on a design module."""

  __slots__ = ["name", "_type"]

  def __init__(self, type: mlir.ir.Type, name: str = None):
    self.name: str = name
    self._type: mlir.ir.Type = type

  @property
  def type(self):
    return self._type


class Output(ModuleDecl):
  pass


class Input(ModuleDecl):
  pass


class Parameter:
  __slots__ = ["name", "attr"]

  def __init__(self, value=None, name: str = None):
    if value is not None:
      self.attr = support.var_to_attribute(value)
    self.name = name


class module:
  """Decorator for module classes or functions which parameterize module
  classes."""

  mod = None
  func = None
  extern_mod = None

  # When the decorator is attached, this runs.
  def __init__(self, func_or_class, extern_name=None):
    self.extern_name = extern_name
    if inspect.isclass(func_or_class):
      # If it's just a module class, we should wrap it immediately
      self.mod = _module_base(func_or_class)
      _register_generator(self.mod.__name__, "extern_instantiate",
                          self._instantiate)
      return
    elif not inspect.isfunction(func_or_class):
      raise TypeError("@module got invalid object")

    # If it's a module parameterization function, inspect the arguments to
    # ensure sanity.
    self.func = func_or_class
    self.sig = inspect.signature(self.func)
    for (_, param) in self.sig.parameters.items():
      if param.kind == param.VAR_KEYWORD:
        raise TypeError("Module parameter definitions cannot have **kwargs")
      if param.kind == param.VAR_POSITIONAL:
        raise TypeError("Module parameter definitions cannot have *args")

  # This function gets executed in two situations:
  #   - In the case of a module function parameterizer, it is called when the
  #   user wants to apply specific parameters to the module. In this case, we
  #   should call the function, wrap the returned module class, and return it.
  #   We _could_ also cache it, though that's not strictly necessary unless the
  #   user is breaking the rules. TODO: cache it (requires all the parameters to
  #   be hashable).
  #   - A simple (non-parameterized) module has been wrapped and the user wants
  #   to construct one. Just forward to the module class' constructor.
  def __call__(self, *args, **kwargs):
    if self.func is not None:
      param_values = self.sig.bind(*args, **kwargs)
      param_values.apply_defaults()
      cls = self.func(*args, **kwargs)
      if cls is None:
        raise ValueError("Parameterization function must return module class")
      mod = _module_base(cls, param_values.arguments)

      if self.extern_name:
        _register_generator(cls.__name__, "extern_instantiate",
                            self._instantiate)
      return mod

    return self.mod(*args, **kwargs)

  # Generator for external modules.
  def _instantiate(self, op):
    # Get the port names from the attributes we stored them in.
    op_names_attrs = mlir.ir.ArrayAttr(op.attributes["opNames"])
    op_names = [mlir.ir.StringAttr(x) for x in op_names_attrs]
    input_ports = [(n.value, o.type) for (n, o) in zip(op_names, op.operands)]

    if self.extern_mod is None:
      # Find the top MLIR module.
      mod = op
      while mod.name != "module":
        mod = mod.parent

      result_names_attrs = mlir.ir.ArrayAttr(op.attributes["resultNames"])
      result_names = [mlir.ir.StringAttr(x) for x in result_names_attrs]
      output_ports = [
          (n.value, o.type) for (n, o) in zip(result_names, op.results)
      ]

      with mlir.ir.InsertionPoint(mod.regions[0].blocks[0]):
        self.extern_mod = hw.HWModuleExternOp(self.extern_name, input_ports,
                                              output_ports)

    attrs = {
        nattr.name: nattr.attr
        for nattr in op.attributes
        if nattr.name not in ["opNames", "resultNames"]
    }

    with mlir.ir.InsertionPoint(op):
      mapping = {name.value: op.operands[i] for i, name in enumerate(op_names)}
      inst = self.extern_mod.create(op.name, **mapping).operation
      for (name, attr) in attrs.items():
        inst.attributes[name] = attr
      return inst


def externmodule(cls_or_name):
  """Wrap an externally implemented module. If no name given in the decorator
  argument, use the class name."""

  if isinstance(cls_or_name, str):
    return lambda cls: module(cls, cls_or_name)
  return module(cls_or_name, cls_or_name.__name__)


# The real workhorse of this package. Wraps a module class, making it implement
# MLIR's OpView parent class.
def _module_base(cls, params={}):
  """The CIRCT design entry module class decorator."""

  class mod(cls, mlir.ir.OpView):

    # Default mappings to operand/result numbers.
    _input_ports: list[(str, mlir.ir.Type)] = []
    _output_ports: list[(str, mlir.ir.Type)] = []
    _parameters: dict[str, mlir.ir.PyAttribute] = {}

    def __init__(self, *args, **kwargs):
      """Scan the class and eventually instance for Input/Output members and
      treat the inputs as operands and outputs as results."""

      inputs = {
          name: kwargs[name] for (name, _) in mod._input_ports if name in kwargs
      }
      pass_up_kwargs = {n: v for (n, v) in kwargs.items() if n not in inputs}
      # TODO: Inspect signature, raise error if __init__ doesn't accept **kwargs
      cls.__init__(self, *args, **pass_up_kwargs)

      # Build a list of operand values for the operation we're gonna create.
      input_ports_values: list[mlir.ir.Value] = []
      self.backedges: dict[int:BackedgeBuilder.Edge] = {}
      for (idx, (name, type)) in enumerate(mod._input_ports):
        if name in inputs:
          value = support.get_value(inputs[name])
          assert value is not None
        else:
          backedge = BackedgeBuilder.current().create(type, name, self)
          self.backedges[idx] = backedge
          value = backedge.result
        input_ports_values.append(value)

      # Set up the op attributes.
      attributes: dict[str:mlir.ir.Attribute] = {}
      for attr_name in dir(self):
        if attr_name.startswith('_') or attr_name in cls._dont_touch:
          continue
        attr = getattr(self, attr_name)
        mlir_attr = support.var_to_attribute(attr, True)
        if mlir_attr is not None:
          attributes[attr_name] = mlir_attr
      attributes["parameters"] = mlir.ir.DictAttr.get(mod._parameters)

      # Store the port names as attributes.
      attributes["opNames"] = mlir.ir.ArrayAttr.get(
          [mlir.ir.StringAttr.get(name) for (name, _) in mod._input_ports])
      attributes["resultNames"] = mlir.ir.ArrayAttr.get(
          [mlir.ir.StringAttr.get(name) for (name, _) in mod._output_ports])

      # Init the OpView, which creates the operation.
      mlir.ir.OpView.__init__(
          self,
          self.build_generic(attributes=attributes,
                             results=[type for (_, type) in mod._output_ports],
                             operands=input_ports_values))

    @staticmethod
    def inputs() -> list[(str, mlir.ir.Type)]:
      """Return the list of input ports."""
      return mod._input_ports

  mod.__name__ = cls.__name__
  mod.OPERATION_NAME = OPERATION_NAMESPACE + cls.__name__

  # Inputs, Outputs, and parameters are all class members. We must populate
  # them.  First, scan 'cls' for them.
  for attr_name in dir(cls):
    if attr_name.startswith("_"):
      continue
    attr = getattr(cls, attr_name)
    if isinstance(attr, Input):
      attr.name = attr_name
      mod._input_ports.append((attr.name, attr.type))
    elif isinstance(attr, Output):
      attr.name = attr_name
      mod._output_ports.append((attr.name, attr.type))
    elif isinstance(attr, Parameter):
      attr.name = attr_name
      mod._parameters[attr.name] = attr.attr

  # Second, the specified parameters.
  for (name, value) in params.items():
    value = support.var_to_attribute(value, True)
    if value is not None:
      mod._parameters[name] = value
      setattr(mod, name, Parameter(value, name))

  # Third, add the module name, if specified.
  if "get_module_name" in dir(cls):
    mod._parameters["module_name"] = mlir.ir.StringAttr.get(
        mod.get_module_name())

  # Keep a special "don't touch" to skip over as attributes.
  cls._dont_touch = set()
  cls._dont_touch.update(dir(mlir.ir.OpView))
  cls._dont_touch.add("OPERATION_NAME")

  # Add accessors for each input and output port to emulate generated OpView
  # subclasses. Add the names to "don't touch" since they can't be touched
  # (since they implictly call an OpView property) when the attributes are being
  # scanned in the `mod` constructor.
  for (idx, (name, _)) in enumerate(mod._input_ports):
    setattr(
        mod, name,
        property(lambda self, idx=idx: OpOperand(self, idx, self.operands[idx],
                                                 self)))
    cls._dont_touch.add(name)
  mod._input_ports_lookup = dict(mod._input_ports)
  for (idx, (name, type)) in enumerate(mod._output_ports):
    setattr(mod, name, property(lambda self: Value(self.results[idx], type)))
    cls._dont_touch.add(name)
  mod._output_ports_lookup = dict(mod._output_ports)

  _register_generators(mod)
  return mod


def _register_generators(modcls):
  """Scan the class, looking for and registering _Generators."""
  for name in dir(modcls):
    member = getattr(modcls, name)
    if isinstance(member, _Generate):
      member.modcls = modcls
      _register_generator(modcls.__name__, name, member)


def _register_generator(class_name, generator_name, generator):
  circt.msft.register_generator(mlir.ir.Context.current,
                                OPERATION_NAMESPACE + class_name,
                                generator_name, generator)


class _Generate:
  """Represents a generator. Stores the generate function and wraps it with the
  necessary logic to build an HWModule."""

  def __init__(self, gen_func):
    self.gen_func = gen_func
    self.modcls = None

    # Track generated modules so we don't create unnecessary duplicates of
    # modules that are structurally equivalent.
    self.generated_modules = {}

  def __call__(self, op):
    """Build an HWModuleOp and run the generator as the body builder."""

    # Find the top MLIR module.
    mod = op
    while mod.name != "module":
      mod = mod.parent

    # Get the port names from the attributes we stored them in.
    op_names_attrs = mlir.ir.ArrayAttr(op.attributes["opNames"])
    op_names = [mlir.ir.StringAttr(x) for x in op_names_attrs]
    input_ports = [(n.value, o.type) for (n, o) in zip(op_names, op.operands)]

    result_names_attrs = mlir.ir.ArrayAttr(op.attributes["resultNames"])
    result_names = [mlir.ir.StringAttr(x) for x in result_names_attrs]
    output_ports = [
        (n.value, o.type) for (n, o) in zip(result_names, op.results)
    ]

    # Assemble the parameters.
    self.params = {
        nattr.name: support.attribute_to_var(nattr.attr)
        for nattr in mlir.ir.DictAttr(op.attributes["parameters"])
    }

    attrs = {
        nattr.name: nattr.attr
        for nattr in op.attributes
        if nattr.name not in ["opNames", "resultNames", "parameters"]
    }

    # Build the replacement HWModuleOp in the outer module.
    module_key = str((op.name, sorted(input_ports), sorted(output_ports),
                      sorted(self.params.items())))
    if "module_name" in self.params:
      module_name = self.params["module_name"]
    else:
      module_name = op.name
    if module_key not in self.generated_modules:
      with mlir.ir.InsertionPoint(mod.regions[0].blocks[0]):
        gen_mod = ModuleDefinition(self.modcls,
                                   module_name,
                                   input_ports=input_ports,
                                   output_ports=output_ports,
                                   body_builder=self.gen_func)
        self.generated_modules[module_key] = gen_mod

    # Build a replacement instance at the op to be replaced.
    with mlir.ir.InsertionPoint(op):
      mapping = {name.value: op.operands[i] for i, name in enumerate(op_names)}
      inst = self.generated_modules[module_key].create(op.name,
                                                       **mapping).operation
      for (name, attr) in attrs.items():
        inst.attributes[name] = attr
      return inst


def generator(func):
  # Convert the generator function to a _Generate class
  return _Generate(func)


class ModuleDefinition(hw.HWModuleOp):

  def __init__(self,
               modcls,
               name,
               input_ports=[],
               output_ports=[],
               body_builder=None):
    self.modcls = modcls
    super().__init__(name,
                     input_ports=input_ports,
                     output_ports=output_ports,
                     body_builder=body_builder)

  # Support attribute access to block arguments by name
  def __getattr__(self, name):
    if name in self.input_indices:
      index = self.input_indices[name]
      return Value(self.entry_block.arguments[index],
                   self.modcls._input_ports_lookup[name])
    raise AttributeError(f"unknown input port name {name}")
