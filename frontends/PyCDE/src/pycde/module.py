#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Tuple, Union, Dict
import typing

from pycde.support import obj_to_value

from .pycde_types import types
from .support import (get_user_loc, var_to_attribute, OpOperandConnect,
                      create_type_string, create_const_zero)
from .value import Value

from circt import support
from circt.dialects import hw, msft
from circt.support import BackedgeBuilder, attribute_to_var

import mlir.ir

import builtins
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


def _create_module_name(name: str, params: mlir.ir.DictAttr):
  """Create a "reasonable" module name from a base name and a set of
  parameters. E.g. PolyComputeForCoeff_62_42_6."""

  def val_str(val):
    if isinstance(val, mlir.ir.Type):
      return create_type_string(val)
    if isinstance(val, mlir.ir.Attribute):
      return str(attribute_to_var(val))
    return str(val)

  param_strings = []
  for p in params:
    param_strings.append(p.name + val_str(p.attr))
  for ps in sorted(param_strings):
    name += "_" + ps

  ret = ""
  name = name.replace("!hw.", "")
  for c in name:
    if c.isalnum():
      ret = ret + c
    elif c not in "!>[],\"" and len(ret) > 0 and ret[-1] != "_":
      ret = ret + "_"
  return ret.strip("_")


# Two problems with this class:
#   (1) It's not sensitive to the System.
#   (2) It keeps references MLIR ops around.
# Possible solution to both involves using System as storage.
class _SpecializedModule:
  """SpecializedModule serves two purposes:

  (1) As a level of indirection between pure python and python CIRCT op
  classes. This indirection makes it possible to invalidate the reference and
  clean up when those ops may not exist anymore.

  (2) It delays module op creation until there is a valid context and system to
  create it in. As a result of how the delayed creation works, module ops are
  only created if said module is instantiated."""

  __slots__ = [
      "circt_mod", "name", "generators", "modcls", "loc", "input_ports",
      "input_port_lookup", "output_ports", "parameters", "extern_name"
  ]

  def __init__(self, cls: type, parameters: Union[dict, mlir.ir.DictAttr],
               extern_name: str):
    self.modcls = cls
    self.circt_mod = None
    self.extern_name = extern_name
    self.loc = get_user_loc()

    # Make sure 'parameters' is a DictAttr rather than a python value.
    self.parameters = var_to_attribute(parameters)
    assert isinstance(self.parameters, mlir.ir.DictAttr)

    # Get the module name
    if extern_name is not None:
      self.name = extern_name
    elif "module_name" in dir(cls):
      self.name = cls.module_name
    elif "get_module_name" in dir(cls):
      self.name = cls.get_module_name()
    else:
      self.name = _create_module_name(cls.__name__, self.parameters)

    # Inputs, Outputs, and parameters are all class members. We must populate
    # them. Scan 'cls' for them.
    self.input_ports = []
    self.input_port_lookup: Dict[str, int] = {}  # Used by 'BlockArgs' below.
    self.output_ports = []
    self.generators = {}
    for attr_name in dir(cls):
      if attr_name.startswith("_"):
        continue
      attr = getattr(cls, attr_name)

      if isinstance(attr, Input):
        attr.name = attr_name
        self.input_ports.append((attr.name, attr.type))
        self.input_port_lookup[attr_name] = len(self.input_ports) - 1
      elif isinstance(attr, Output):
        attr.name = attr_name
        self.output_ports.append((attr.name, attr.type))
      elif isinstance(attr, _Generate):
        self.generators[attr_name] = attr

  def add_accessors(self):
    """Add accessors for each input and output port to emulate generated OpView
     subclasses."""
    for (idx, (name, type)) in enumerate(self.input_ports):
      setattr(
          self.modcls, name,
          property(lambda self, idx=idx: OpOperandConnect(
              self._instantiation.operation, idx, self._instantiation.operation.
              operands[idx], self)))
    for (idx, (name, type)) in enumerate(self.output_ports):
      setattr(
          self.modcls, name,
          property(lambda self, idx=idx, type=type: Value.get(
              self._instantiation.operation.results[idx], type)))

  # Bug: currently only works with one System. See notes at the top of this
  # class.
  def create(self):
    """Create the module op. Should not be called outside of a 'System'
    context."""
    if self.circt_mod is not None:
      return
    from .system import System
    sys = System.current()
    symbol = sys.create_symbol(self.name)

    if self.extern_name is None:
      self.circt_mod = msft.MSFTModuleOp(symbol,
                                         self.input_ports,
                                         self.output_ports,
                                         self.parameters,
                                         loc=self.loc,
                                         ip=sys._get_ip())
      sys._generate_queue.append(self)
    else:
      paramdecl_list = [
          hw.ParamDeclAttr.get_nodefault(i.name,
                                         mlir.ir.TypeAttr.get(i.attr.type))
          for i in self.parameters
      ]
      self.circt_mod = hw.HWModuleExternOp(
          symbol,
          self.input_ports,
          self.output_ports,
          parameters=paramdecl_list,
          attributes={"verilogName": mlir.ir.StringAttr.get(self.extern_name)},
          loc=self.loc,
          ip=sys._get_ip())
    self.add_accessors()

  @property
  def is_created(self):
    return self.circt_mod is not None

  def instantiate(self, instance_name: str, inputs: dict, loc):
    """Create a instance op."""
    if self.extern_name is None:
      return self.circt_mod.create(instance_name, **inputs, loc=loc)
    else:
      return self.circt_mod.create(instance_name,
                                   **inputs,
                                   parameters=self.parameters,
                                   loc=loc)

  def generate(self):
    """Fill in (generate) this module. Only supports a single generator
    currently."""
    assert len(self.generators) == 1
    for g in self.generators.values():
      g.generate(self)
      return


# Set an input to no_connect to indicate not to connect it. Only valid for
# external module inputs.
no_connect = object()


def module(func_or_class):
  """Decorator to signal that a class should be treated as a module or a
  function should be treated as a module parameterization function. In the
  latter case, the function must return a python class to be treated as the
  parameterized module."""
  if inspect.isclass(func_or_class):
    # If it's just a module class, we should wrap it immediately
    return _module_base(func_or_class, None)
  elif inspect.isfunction(func_or_class):
    return _parameterized_module(func_or_class, None)
  raise TypeError(
      "@module decorator must be on class or parameterization function")


def _get_module_cache_key(func,
                          params) -> Tuple[builtins.function, mlir.ir.DictAttr]:
  """The "module" cache is specifically for parameterized modules. It maps the
  module parameterization function AND parameter values to the class which was
  generated by a previous call to said module parameterization function."""
  if not isinstance(params, mlir.ir.DictAttr):
    params = var_to_attribute(params)
  return (func, mlir.ir.Attribute(params))


# A memoization table for module parameterization function calls.
_module_cache: typing.Dict[Tuple[builtins.function, mlir.ir.DictAttr],
                           object] = {}


class _parameterized_module:
  """When the @module decorator detects that it is decorating a function, use
  this class to wrap it."""

  mod = None
  func = None
  extern_mod = None

  # When the decorator is attached, this runs.
  def __init__(self, func: builtins.function, extern_name):
    self.extern_name = extern_name

    # If it's a module parameterization function, inspect the arguments to
    # ensure sanity.
    self.func = func
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
    assert self.func is not None
    param_values = self.sig.bind(*args, **kwargs)
    param_values.apply_defaults()

    # Function arguments which start with '_' don't become parameters.
    params = {
        n: v for n, v in param_values.arguments.items() if not n.startswith("_")
    }

    # Check cache
    cache_key = _get_module_cache_key(self.func, params)
    if cache_key in _module_cache:
      return _module_cache[cache_key]

    cls = self.func(*args, **kwargs)
    if cls is None:
      raise ValueError("Parameterization function must return module class")

    mod = _module_base(cls, self.extern_name, params)
    _module_cache[cache_key] = mod
    return mod


def externmodule(to_be_wrapped, extern_name=None):
  """Wrap an externally implemented module. If no name given in the decorator
  argument, use the class name."""

  if isinstance(to_be_wrapped, str):
    return lambda cls, extern_name=to_be_wrapped: externmodule(cls, extern_name)

  if extern_name is None:
    extern_name = to_be_wrapped.__name__
  if inspect.isclass(to_be_wrapped):
    # If it's just a module class, we should wrap it immediately
    return _module_base(to_be_wrapped, extern_name)
  return _parameterized_module(to_be_wrapped, extern_name)


def _module_base(cls, extern_name: str, params={}):
  """Wrap a class, making it a PyCDE module."""

  class mod(cls):
    __name__ = cls.__name__
    _pycde_mod = _SpecializedModule(cls, params, extern_name)

    def __init__(self, *args, **kwargs):
      """Scan the class and eventually instance for Input/Output members and
      treat the inputs as operands and outputs as results."""
      # Ensure the module has been created.
      mod._pycde_mod.create()
      # Get the user callsite.
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
      for (idx, (name, type)) in enumerate(mod._pycde_mod.input_ports):
        if name in inputs:
          input = inputs[name]
          if input == no_connect:
            if extern_name is None:
              raise ConnectionError(
                  "`no_connect` is only valid on extern module ports")
            else:
              value = Value.get(create_const_zero(type))
          else:
            value = obj_to_value(input, type)
        else:
          backedge = BackedgeBuilder.current().create(type,
                                                      name,
                                                      mod._pycde_mod.circt_mod,
                                                      loc=loc)
          self.backedges[idx] = backedge
          value = Value.get(backedge.result)
        inputs[name] = value

      instance_name = cls.__name__
      if "instance_name" in dir(self):
        instance_name = self.instance_name
      self._instantiation = mod._pycde_mod.instantiate(instance_name,
                                                       inputs,
                                                       loc=loc)

    @staticmethod
    def inputs() -> list[(str, mlir.ir.Type)]:
      """Return the list of input ports."""
      return mod._pycde_mod.input_ports

    @staticmethod
    def outputs() -> list[(str, mlir.ir.Type)]:
      """Return the list of input ports."""
      return mod._pycde_mod.output_ports

  return mod


class _Generate:
  """Represents a generator. Stores the generate function and wraps it with the
  necessary logic to build a module."""

  def __init__(self, gen_func):
    sig = inspect.signature(gen_func)
    if len(sig.parameters) != 1:
      raise ValueError(
          "Generate functions must take one argument and do not support 'self'."
      )
    self.gen_func = gen_func
    self.loc = get_user_loc()

  def generate(self, specialized_mod: _SpecializedModule):
    """Build an HWModuleOp and run the generator as the body builder."""

    entry_block = specialized_mod.circt_mod.add_entry_block()
    with mlir.ir.InsertionPoint(entry_block), self.loc, BackedgeBuilder():
      args = BlockArgs(specialized_mod)
      outputs = self.gen_func(args)
      self._create_output_op(outputs, specialized_mod)

  def _create_output_op(self, gen_ret, modcls):
    """Create the hw.OutputOp from the generator returns."""
    output_ports = modcls.output_ports

    # If generator didn't return anything, this op mustn't have any outputs.
    if gen_ret is None:
      if len(output_ports) == 0:
        msft.OutputOp([])
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

    msft.OutputOp(outputs)


def generator(func):
  """Decorator for generation functions."""
  return _Generate(func)


class BlockArgs:
  """Get the input ports."""

  def __init__(self, mod: _SpecializedModule):
    self.mod = mod

  # Support attribute access to block arguments by name
  def __getattr__(self, name):
    if name not in self.mod.input_port_lookup:
      raise AttributeError(f"unknown input port name {name}")
    idx = self.mod.input_port_lookup[name]
    val = self.mod.circt_mod.entry_block.arguments[idx]
    return Value.get(val)
