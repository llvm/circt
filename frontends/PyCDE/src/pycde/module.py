#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Union

from pycde.support import obj_to_value

from .pycde_types import types
from .support import (get_user_loc, var_to_attribute, OpOperandConnect,
                      create_type_string)
from .value import Value

from circt import support
from circt.dialects import hw, msft
from circt.support import BackedgeBuilder
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


class _SpecializedModule:
  __slots__ = [
      "circt_mod", "generators", "modcls", "input_ports", "output_ports"
  ]

  def __init__(self, cls: type, parameters: Union[dict, mlir.ir.DictAttr],
               extern: bool):
    self.modcls = cls
    loc = get_user_loc()
    # Get the module name
    if "module_name" in dir(cls) and isinstance(cls.module_name, str):
      name = cls.module_name
    else:
      name = cls.__name__

    # Make sure 'parameters' is a DictAttr rather than a python value.
    parameters = var_to_attribute(parameters)
    assert isinstance(parameters, mlir.ir.DictAttr)

    # Inputs, Outputs, and parameters are all class members. We must populate
    # them.  First, scan 'cls' for them.
    self.input_ports = []
    self.output_ports = []
    for attr_name in dir(cls):
      if attr_name.startswith("_"):
        continue
      attr = getattr(cls, attr_name)

      if isinstance(attr, Input):
        attr.name = attr_name
        self.input_ports.append((attr.name, attr.type))
      elif isinstance(attr, Output):
        attr.name = attr_name
        self.output_ports.append((attr.name, attr.type))

    if not extern:
      self.circt_mod = msft.MSFTModuleOp(name,
                                         self.input_ports,
                                         self.output_ports,
                                         parameters,
                                         loc=loc)
    else:
      self.circt_mod = hw.HWModuleExternOp(
          name,
          self.input_ports,
          self.output_ports,
          attributes={"parameters": parameters},
          loc=loc)
    self.add_accessors()
    self.generators = {}

  def add_accessors(self):
    """Add accessors for each input and output port to emulate generated OpView
     subclasses."""
    for (idx, (name, type)) in enumerate(self.input_ports):
      setattr(
          self.modcls, name,
          property(lambda self, idx=idx: OpOperandConnect(
              self, idx, self._instantiation.operands[idx], self)))
    for (idx, (name, type)) in enumerate(self.output_ports):
      setattr(
          self.modcls, name,
          property(lambda self, idx=idx, type=type: Value.get(
              self._instantiation.results[idx], type)))


class Parameter:
  __slots__ = ["name", "attr"]

  def __init__(self, value=None, name: str = None):
    if value is not None:
      self.attr = var_to_attribute(value)
    self.name = name


# Set an input to no_connect to indicate not to connect it. Only valid for
# external module inputs.
no_connect = object()


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
      self.mod = _module_base(func_or_class, extern_name is not None)
      return
    elif not inspect.isfunction(func_or_class):
      raise TypeError(
          "@module decorator must be on class or parameterization function")

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

      # Function arguments which start with '_' don't become parameters.
      params = {
          n: v
          for n, v in param_values.arguments.items()
          if not n.startswith("_")
      }
      return _module_base(cls, self.extern_name is not None, params)

      # if self.extern_name:
      #   _register_generator(cls.__name__, "extern_instantiate",
      #                       self._instantiate,
      #                       mlir.ir.DictAttr.get(mod._parameters))
      # return mod

    return self.mod(*args, **kwargs)

  # # Generator for external modules.
  # def _instantiate(self, op):
  #   # Get the port names from the attributes we stored them in.
  #   op_names_attrs = mlir.ir.ArrayAttr(op.attributes["opNames"])
  #   op_names = [mlir.ir.StringAttr(x) for x in op_names_attrs]

  #   if self.extern_mod is None:
  #     # Find the top MLIR module.
  #     mod = op
  #     while mod.parent is not None:
  #       mod = mod.parent

  #     input_ports = [(n.value, o.type) for (n, o) in zip(op_names, op.operands)]
  #     result_names_attrs = mlir.ir.ArrayAttr(op.attributes["resultNames"])
  #     result_names = [mlir.ir.StringAttr(x) for x in result_names_attrs]
  #     output_ports = [
  #         (n.value, o.type) for (n, o) in zip(result_names, op.results)
  #     ]

  #     with mlir.ir.InsertionPoint(mod.regions[0].blocks[0]):
  #       self.extern_mod = hw.HWModuleExternOp(self.extern_name, input_ports,
  #                                             output_ports)

  #   attrs = {
  #       nattr.name: nattr.attr
  #       for nattr in op.attributes
  #       if nattr.name not in ["opNames", "resultNames"]
  #   }

  #   with mlir.ir.InsertionPoint(op):
  #     mapping = {name.value: op.operands[i] for i, name in enumerate(op_names)}
  #     result_types = [x.type for x in op.results]
  #     inst = self.extern_mod.create(op.name, **mapping,
  #                                   results=result_types).operation
  #     for (name, attr) in attrs.items():
  #       inst.attributes[name] = attr
  #     return inst


def externmodule(cls_or_name):
  """Wrap an externally implemented module. If no name given in the decorator
  argument, use the class name."""

  if isinstance(cls_or_name, str):
    return lambda cls: module(cls, cls_or_name)
  return module(cls_or_name, cls_or_name.__name__)


def _module_base(cls, extern: bool, params={}):
  """Wrap a class, making it a PyCDE module."""

  class mod(cls):
    __name__ = cls.__name__
    _pycde_mod = _SpecializedModule(cls, params, extern)

    def __init__(self, *args, **kwargs):
      """Scan the class and eventually instance for Input/Output members and
      treat the inputs as operands and outputs as results."""

      # Get the user callsite
      loc = get_user_loc()

      inputs = {
          name: kwargs[name]
          for (name, _) in mod._pycde_mod.input_ports
          if name in kwargs
      }
      pass_up_kwargs = {n: v for (n, v) in kwargs.items() if n not in inputs}
      if len(pass_up_kwargs) > 0:
        init_sig = inspect.signature(cls.__init__)
        if not any(
            [x == inspect.Parameter.VAR_KEYWORD for x in init_sig.parameters]):
          raise ValueError("Module constructor doesn't have a **kwargs"
                           " parameter, so the following are likely inputs"
                           " which don't have a port: " +
                           ",".join(pass_up_kwargs.keys()))
      cls.__init__(self, *args, **pass_up_kwargs)

      # Build a list of operand values for the operation we're gonna create.
      self.backedges: dict[str:BackedgeBuilder.Edge] = {}
      for (name, type) in mod._pycde_mod.input_ports:
        if name in inputs:
          input = inputs[name]
          if input == no_connect:
            if not extern:
              raise ConnectionError(
                  "`no_connect` is only valid on extern module ports")
            else:
              value = Value.get(hw.ConstantOp.create(types.i1, 0).result)
          else:
            value = obj_to_value(input, type)
        else:
          backedge = BackedgeBuilder.current().create(type, name, self, loc=loc)
          self.backedges[name] = backedge
          value = Value.get(backedge.result)
        inputs[name] = value

      self._instantiation = mod._pycde_mod.circt_mod.create("", inputs, loc=loc)

    # def output_values(self):
    #   return {
    #       op_name: self.operation.results[idx]
    #       for (idx, (op_name, _)) in enumerate(mod._output_ports)
    #   }

    @staticmethod
    def inputs() -> list[(str, mlir.ir.Type)]:
      """Return the list of input ports."""
      return mod._pycde_mod.input_ports

    @staticmethod
    def outputs() -> list[(str, mlir.ir.Type)]:
      """Return the list of input ports."""
      return mod._pycde_mod.output_ports

  return mod


def _register_generators(modcls, parameters: mlir.ir.Attribute):
  """Scan the class, looking for and registering _Generators."""
  for name in dir(modcls):
    member = getattr(modcls, name)
    if isinstance(member, _Generate):
      member.modcls = modcls
      _register_generator(modcls.__name__, name, member, parameters)


def _register_generator(class_name, generator_name, generator, parameters):
  circt.msft.register_generator(mlir.ir.Context.current,
                                OPERATION_NAMESPACE + class_name,
                                generator_name, generator, parameters)


class _Generate:
  """Represents a generator. Stores the generate function and wraps it with the
  necessary logic to build an HWModule."""

  def __init__(self, gen_func):
    sig = inspect.signature(gen_func)
    if len(sig.parameters) != 1:
      raise ValueError(
          "Generate functions must take one argument and do not support 'self'."
      )
    self.gen_func = gen_func
    self.modcls = None
    self.loc = get_user_loc()

  def _generate(self, mod):
    outputs = self.gen_func(mod)
    self._create_output_op(outputs)

  def __call__(self, op):
    """Build an HWModuleOp and run the generator as the body builder."""

    # Find the top MLIR module.
    mod = op
    while mod.parent is not None:
      mod = mod.parent

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
    if "module_name" in self.params:
      module_name = self.params["module_name"]
    else:
      module_name = self.create_module_name(op)
      self.params["module_name"] = module_name
    module_name = self.sanitize(module_name)

    # Track generated modules so we don't create unnecessary duplicates of
    # modules that are structurally equivalent. If the module name exists in the
    # top level MLIR module, assume that we've already generated it.
    existing_module_names = [
        o for o in mod.regions[0].blocks[0].operations
        if mlir.ir.StringAttr(o.name).value == module_name
    ]

    if not existing_module_names:
      with mlir.ir.InsertionPoint(mod.regions[0].blocks[0]), self.loc:
        mod = ModuleDefinition(self.modcls,
                               module_name,
                               input_ports=self.modcls._input_ports,
                               output_ports=self.modcls._output_ports,
                               body_builder=self._generate)
    else:
      assert (len(existing_module_names) == 1)
      mod = existing_module_names[0]

    # Build a replacement instance at the op to be replaced.
    op_names = [name for name, _ in self.modcls._input_ports]
    with mlir.ir.InsertionPoint(op):
      mapping = {name: op.operands[i] for i, name in enumerate(op_names)}
      inst = mod.create(op.name, **mapping).operation
      for (name, attr) in attrs.items():
        if name == "parameters":
          continue
        inst.attributes[name] = attr
      return inst

  def create_module_name(self, op):

    def val_str(val):
      if isinstance(val, mlir.ir.Type):
        return create_type_string(val)
      return str(val)

    name = op.name
    if len(self.params) > 0:
      name += "_" + "_".join(
          val_str(value) for (_, value) in sorted(self.params.items()))

    return name

  def sanitize(self, value):
    sanitized_str = str(value)
    for sub in ["!hw.", ">", "[", "]", ","]:
      sanitized_str = sanitized_str.replace(sub, "")
    for sub in ["<", "x", " "]:
      sanitized_str = sanitized_str.replace(sub, "_")
    return sanitized_str

  def _create_output_op(self, gen_ret):
    """Create the hw.OutputOp from the generator returns."""
    assert hasattr(self, "modcls")
    output_ports = self.modcls._output_ports

    # If generator didn't return anything, this op mustn't have any outputs.
    if gen_ret is None:
      if len(output_ports) == 0:
        hw.OutputOp([])
        return
      raise support.ConnectionError("Generator must return dict")

    # Now create the output op depending on the object type returned
    outputs: list[Value] = list()

    # Only acceptable return is a dict of port, value mappings.
    if not isinstance(gen_ret, dict):
      raise support.ConnectionError("Generator must return a dict of outputs")

    # A dict of `OutputPortName` -> ValueLike or convertable objects must be
    # converted to a list in port order.
    unconnected_ports = []
    for (name, port_type) in output_ports:
      if name not in gen_ret:
        unconnected_ports.append(name)
        outputs.append(None)
      else:
        val = obj_to_value(gen_ret[name], port_type).value
        outputs.append(val)
        gen_ret.pop(name)
    if len(unconnected_ports) > 0:
      raise support.UnconnectedSignalError(unconnected_ports)
    if len(gen_ret) > 0:
      raise support.ConnectionError(
          "Could not map the following to output ports: " +
          ",".join(gen_ret.keys()))

    hw.OutputOp(outputs)


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
      val = self.entry_block.arguments[index]
      if self.modcls:
        ty = self.modcls._input_ports_lookup[name]
      else:
        ty = support.type_to_pytype(val.type)
      return Value.get(val, ty)
    raise AttributeError(f"unknown input port name {name}")
