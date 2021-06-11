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

  def __init__(self, value, name: str = None):
    self.attr = support.var_to_attribute(value)
    self.name = name


def module(cls):
  """The CIRCT design entry module class decorator."""

  class __Module(cls, mlir.ir.OpView):
    OPERATION_NAME = OPERATION_NAMESPACE + cls.__name__
    _ODS_REGIONS = (0, True)

    # Default mappings to operand/result numbers.
    input_ports: dict[str, int] = {}
    output_ports: dict[str, int] = {}

    def __init__(self, *args, inputs={}, **kwargs):
      """Scan the class and eventually instance for Input/Output members and
      treat the inputs as operands and outputs as results."""
      cls.__init__(self, *args, **kwargs)

      # The OpView attributes cannot be touched before OpView is constructed.
      # Get a list and don't touch them.
      dont_touch = [x for x in dir(mlir.ir.OpView)]
      dont_touch.append("OPERATION_NAME")

      # After the wrapped class' construct, all the IO should be known.
      input_ports: list[Input] = []
      output_ports: list[Output] = []
      parameters: list[Parameter] = []
      attributes: dict[str:mlir.ir.Attribute] = {}
      # Scan for them.
      for attr_name in dir(self):
        if attr_name in dont_touch or attr_name.startswith("_"):
          continue
        attr = self.__getattribute__(attr_name)
        if isinstance(attr, Input):
          attr.name = attr_name
          input_ports.append(attr)
        elif isinstance(attr, Output):
          attr.name = attr_name
          output_ports.append(attr)
        elif isinstance(attr, Parameter):
          attr.name = attr_name
          parameters.append(attr)
        else:
          mlir_attr = support.var_to_attribute(attr, True)
          if mlir_attr is not None:
            attributes[attr_name] = mlir_attr

      # Build a list of operand values for the operation we're gonna create.
      input_ports_values: list[mlir.ir.Value] = []
      self.backedges: dict[int:BackedgeBuilder.Edge] = {}
      for (idx, input) in enumerate(input_ports):
        if input.name in inputs:
          value = inputs[input.name]
          if isinstance(value, mlir.ir.OpView):
            value = value.operation.result
          elif isinstance(value, mlir.ir.Operation):
            value = value.result
          assert isinstance(value, mlir.ir.Value)
        else:
          backedge = BackedgeBuilder.current().create(input.type, input.name,
                                                      self)
          self.backedges[idx] = backedge
          value = backedge.result
        input_ports_values.append(value)

      # Store the port names as attributes.
      op_names_attr = mlir.ir.ArrayAttr.get(
          [mlir.ir.StringAttr.get(x.name) for x in input_ports])
      result_names_attr = mlir.ir.ArrayAttr.get(
          [mlir.ir.StringAttr.get(x.name) for x in output_ports])

      # Set up the op attributes.
      parameters = {p.name: p.attr for p in parameters}
      attributes["parameters"] = mlir.ir.DictAttr.get(parameters)
      attributes["opNames"] = op_names_attr
      attributes["resultNames"] = result_names_attr

      # Init the OpView, which creates the operation.
      mlir.ir.OpView.__init__(
          self,
          self.build_generic(attributes=attributes,
                             results=[x.type for x in output_ports],
                             operands=[x for x in input_ports_values]))

      # Build the mappings for __getattribute__.
      self.input_ports = {port.name: i for i, port in enumerate(input_ports)}
      self.output_ports = {port.name: i for i, port in enumerate(output_ports)}

    def set_module_name(self, name):
      """Set the name of the generated module. Must be used when more than one
      module is generated as a result of parameterization. Must be unique."""
      self.module_name = Parameter(name)

    def __getattribute__(self, name: str):
      # Base case.
      if name == "input_ports" or name == "output_ports" or \
         name == "operands" or name == "results":
        return super().__getattribute__(name)

      # To emulate OpView, if 'name' is either an input or output port,
      # redirect.
      if name in self.input_ports:
        op_num = self.input_ports[name]
        operand = self.operands[op_num]
        return OpOperand(self, op_num, operand, self)
      if name in self.output_ports:
        op_num = self.output_ports[name]
        return self.results[op_num]
      return super().__getattribute__(name)

    @staticmethod
    def inputs() -> dict[str:mlir.ir.Type]:
      input_ports: dict[str:mlir.ir.Type] = {}
      for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if isinstance(attr, Input):
          input_ports[attr_name] = attr.type
      return input_ports

  _register_generators(cls)
  return __Module


def _register_generators(cls):
  """Scan the class, looking for and registering _Generators."""
  for (name, member) in cls.__dict__.items():
    if isinstance(member, _Generate):
      _register_generator(cls.__name__, name, member)


def _register_generator(class_name, generator_name, generator):
  circt.msft.register_generator(mlir.ir.Context.current,
                                OPERATION_NAMESPACE + class_name,
                                generator_name, generator)


def _externmodule(cls, module_name: str):

  class ExternModule(module(cls)):
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

  def _generate(self, mod):
    gf_args = inspect.getfullargspec(self.gen_func).args
    call_args = {}
    for argname in gf_args[1:]:
      if argname in self.params:
        call_args[argname] = self.params[argname]
      else:
        raise ValueError("Cannot find parameter requested by generator func "
                         f"args: {argname}")
    return self.gen_func(mod, **call_args)

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
                                               body_builder=self._generate)
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
