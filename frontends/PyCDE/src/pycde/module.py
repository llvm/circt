#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

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


class modparam:

  def __init__(self, func):
    self.func = func
    self.sig = inspect.signature(func)
    for (name, param) in self.sig.parameters.items():
      if param.kind == param.VAR_KEYWORD:
        raise TypeError("Module parameter definitions cannot have **kwargs")
      if param.kind == param.VAR_POSITIONAL:
        raise TypeError("Module parameter definitions cannot have *args")

  def __call__(self, *args, **kwargs):
    param_values = self.sig.bind(*args, **kwargs)
    param_values.apply_defaults()
    cls = self.func(*args, **kwargs)
    cls = _module_base(cls, param_values.arguments)
    _register_generators(cls)
    return cls


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
  mod._ODS_REGIONS = (0, True)

  # Inputs, Outputs, and parameters are all class members. We must populate them.
  # First, scan 'cls' for them.
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
    setattr(mod, name, property(lambda self: self.operands[idx]))
    cls._dont_touch.add(name)
  for (idx, (name, _)) in enumerate(mod._output_ports):
    setattr(mod, name, property(lambda self: self.results[idx]))
    cls._dont_touch.add(name)

  _register_generators(cls)
  return mod


def module(cls):
  mod = _module_base(cls)
  return mod


def _register_generators(cls):
  """Scan the class, looking for and registering _Generators."""
  for name in dir(cls):
    member = getattr(cls, name)
    if isinstance(member, _Generate):
      _register_generator(cls.__name__, name, member)


def _register_generator(class_name, generator_name, generator):
  circt.msft.register_generator(mlir.ir.Context.current,
                                OPERATION_NAMESPACE + class_name,
                                generator_name, generator)


def _externmodule(cls, module_name: str):

  mod = _module_base(cls)

  class ExternModule(mod):
    _extern_mod = None

    @staticmethod
    def _instantiate(op):
      # Get the port names from the attributes we stored them in.
      op_names_attrs = mlir.ir.ArrayAttr(op.attributes["opNames"])
      op_names = [mlir.ir.StringAttr(x) for x in op_names_attrs]
      input_ports = [(n.value, o.type) for (n, o) in zip(op_names, op.operands)]

      if ExternModule._extern_mod is None:
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
          ExternModule._extern_mod = hw.HWModuleExternOp(
              module_name, input_ports, output_ports)

      attrs = {
          nattr.name: nattr.attr
          for nattr in op.attributes
          if nattr.name not in ["opNames", "resultNames"]
      }

      with mlir.ir.InsertionPoint(op):
        mapping = {
            name.value: op.operands[i] for i, name in enumerate(op_names)
        }
        inst = ExternModule._extern_mod.create(op.name, **mapping).operation
        for (name, attr) in attrs.items():
          inst.attributes[name] = attr
        return inst

  _register_generator(cls.__name__, "extern_instantiate",
                      ExternModule._instantiate)
  ExternModule.__name__ = cls.__name__
  return ExternModule


def externmodule(cls_or_name):
  if isinstance(cls_or_name, str):
    return lambda cls: _externmodule(cls, cls_or_name)
  return _externmodule(cls_or_name, cls_or_name.__name__)


class _Generate:
  """Represents a generator. Stores the generate function and wraps it with the
  necessary logic to build an HWModule."""

  def __init__(self, gen_func):
    self.gen_func = gen_func

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
        gen_mod = circt.dialects.hw.HWModuleOp(module_name,
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


def connect(destination, source):
  if not isinstance(destination, OpOperand):
    raise TypeError(
        f"cannot connect to destination of type {type(destination)}")
  value = support.get_value(source)
  if value is None:
    raise TypeError(f"cannot connect from source of type {type(source)}")

  index = destination.index
  destination.operation.operands[index] = value
  if isinstance(destination, OpOperand) and \
     index in destination.builder.backedges:
    destination.builder.backedges[index].erase()
