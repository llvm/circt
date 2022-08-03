#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Tuple, Union, Dict
from pycde.pycde_types import ClockType

from pycde.support import _obj_to_value

from .common import (AppID, Clock, Input, Output, _PyProxy)
from .support import (get_user_loc, _obj_to_attribute, OpOperandConnect,
                      create_type_string, create_const_zero)
from .value import ClockValue, Value

from circt import support
from circt.dialects import hw, msft
from circt.support import BackedgeBuilder, attribute_to_var

import mlir.ir

import builtins
from contextvars import ContextVar
import inspect
import sys

# A memoization table for module parameterization function calls.
_MODULE_CACHE: Dict[Tuple[builtins.function, mlir.ir.DictAttr], object] = {}


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


def create_msft_module_op(sys, mod: _SpecializedModule, symbol):
  """Creation callback for creating a MSFTModuleOp."""
  return msft.MSFTModuleOp(symbol,
                           mod.input_ports,
                           mod.output_ports,
                           mod.parameters,
                           loc=mod.loc,
                           ip=sys._get_ip())


def generate_msft_module_op(generator: Generator, spec_mod: _SpecializedModule):

  def create_output_op(args: _GeneratorPortAccess):
    """Create the hw.OutputOp from module I/O ports in 'args'."""
    output_ports = spec_mod.output_ports
    outputs: list[Value] = list()

    unconnected_ports = []
    for (name, _) in output_ports:
      if name not in args._output_values:
        unconnected_ports.append(name)
        outputs.append(None)
      else:
        outputs.append(args._output_values[name])
    if len(unconnected_ports) > 0:
      raise support.UnconnectedSignalError(spec_mod.name, unconnected_ports)

    msft.OutputOp([o.value for o in outputs])

  bc = _BlockContext()
  entry_block = spec_mod.circt_mod.add_entry_block()
  with mlir.ir.InsertionPoint(
      entry_block), generator.loc, BackedgeBuilder(), bc:
    args = _GeneratorPortAccess(spec_mod)

    # Enter clock block implicitly if only one clock given
    clk = None
    if len(spec_mod.clock_ports) == 1:
      clk_port = list(spec_mod.clock_ports.values())[0]
      val = entry_block.arguments[clk_port]
      clk = ClockValue(val, ClockType())
      clk.__enter__()

    outputs = generator.gen_func(args)
    if outputs is not None:
      raise ValueError("Generators must not return a value")
    create_output_op(args)

    if clk is not None:
      clk.__exit__(None, None, None)


def create_msft_module_extern_op(sys, mod: _SpecializedModule, symbol):
  """Creation callback for creating a MSFTModuleExternOp."""
  paramdecl_list = [
      hw.ParamDeclAttr.get_nodefault(i.name, mlir.ir.TypeAttr.get(i.attr.type))
      for i in mod.parameters
  ]
  return msft.MSFTModuleExternOp(
      symbol,
      mod.input_ports,
      mod.output_ports,
      parameters=paramdecl_list,
      attributes={"verilogName": mlir.ir.StringAttr.get(mod.extern_name)},
      loc=mod.loc,
      ip=sys._get_ip())


class _SpecializedModule(_PyProxy):
  """SpecializedModule serves two purposes:

  (1) As a level of indirection between pure python and python CIRCT op
  classes. This indirection makes it possible to invalidate the reference and
  clean up when those ops may not exist anymore.

  (2) It delays module op creation until there is a valid context and system to
  create it in. As a result of how the delayed creation works, module ops are
  only created if said module is instantiated."""

  __slots__ = [
      "generators", "modcls", "loc", "clock_ports", "input_ports",
      "input_port_lookup", "output_ports", "output_port_lookup", "parameters",
      "extern_name", "create_cb", "generator_cb"
  ]

  def __init__(self,
               cls: type,
               parameters: Union[dict, mlir.ir.DictAttr],
               extern_name: str,
               create_cb: builtins.function,
               generator_cb: builtins.function = None):
    self.modcls = cls
    self.extern_name = extern_name
    self.loc = get_user_loc()
    self.create_cb = create_cb
    if not self.create_cb:
      raise NotImplementedError(
          "Creation callback function must be set for module")
    self.generator_cb = generator_cb

    # Make sure 'parameters' is a DictAttr rather than a python value.
    self.parameters = _obj_to_attribute(parameters)
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
    self.output_port_lookup: Dict[str, int] = {}  # Used by 'BlockArgs' below.
    self.output_ports = []
    self.generators = {}
    self.clock_ports: Dict[str, int] = {}
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
        self.output_port_lookup[attr_name] = len(self.output_ports) - 1
      elif isinstance(attr, Generator):
        self.generators[attr_name] = attr

      if isinstance(attr, Clock):
        self.clock_ports[attr.name] = len(self.input_ports) - 1
    self.add_accessors()

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
          property(lambda self, idx=idx, type=type: Value(
              self._instantiation.operation.results[idx], type)))

  # Bug: currently only works with one System. See notes at the top of this
  # class.
  def create(self):
    """Create the module op. Should not be called outside of a 'System'
    context. Returns the symbol of the module op."""

    from .system import System
    sys = System.current()
    sys._create_circt_mod(self, self.create_cb)

  @property
  def is_created(self):
    return self.circt_mod is not None

  @property
  def circt_mod(self):
    from .system import System
    sys = System.current()
    return sys._op_cache.get_circt_mod(self)

  def instantiate(self, instance_name: str, inputs: dict, appid: AppID, loc):
    """Create a instance op."""
    if self.extern_name is None:
      ret = self.circt_mod.instantiate(instance_name, **inputs, loc=loc)
    else:
      ret = self.circt_mod.instantiate(instance_name,
                                       **inputs,
                                       parameters=self.parameters,
                                       loc=loc)
    if appid is not None:
      ret.operation.attributes[AppID.AttributeName] = appid._appid
    return ret

  def generate(self):
    """Fill in (generate) this module. Only supports a single generator
    currently."""
    assert len(self.generators) == 1
    for g in self.generators.values():
      self.generator_cb(g, self)
      return

  def print(self, out):
    print(f"<pycde.Module: {self.name} inputs: {self.input_ports} " +
          f"outputs: {self.output_ports}>",
          file=out)


# Set an input to no_connect to indicate not to connect it. Only valid for
# external module inputs.
no_connect = object()


def module(func_or_class):
  """Decorator to signal that a class should be treated as a module or a
  function should be treated as a module parameterization function. In the
  latter case, the function must return a python class to be treated as the
  parameterized module."""
  generate_cb = func_or_class.generator_cb if hasattr(
      func_or_class, "generator_cb") else generate_msft_module_op
  create_cb = func_or_class.create_cb if hasattr(
      func_or_class, "create_cb") else create_msft_module_op
  if inspect.isclass(func_or_class):
    # If it's just a module class, we should wrap it immediately
    return _module_base(func_or_class,
                        None,
                        generator_cb=generate_cb,
                        create_cb=create_cb)
  elif inspect.isfunction(func_or_class):
    return _parameterized_module(func_or_class,
                                 None,
                                 generator_cb=generate_cb,
                                 create_cb=create_cb)
  raise TypeError(
      "@module decorator must be on class or parameterization function")


def _get_module_cache_key(func,
                          params) -> Tuple[builtins.function, mlir.ir.DictAttr]:
  """The "module" cache is specifically for parameterized modules. It maps the
  module parameterization function AND parameter values to the class which was
  generated by a previous call to said module parameterization function."""
  if not isinstance(params, mlir.ir.DictAttr):
    params = _obj_to_attribute(params)
  return (func, mlir.ir.Attribute(params))


class _parameterized_module:
  """When the @module decorator detects that it is decorating a function, use
  this class to wrap it."""

  mod = None
  func = None
  extern_mod = None

  # When the decorator is attached, this runs.
  def __init__(self,
               func: builtins.function,
               extern_name,
               create_cb: builtins.function,
               generator_cb: builtins.function = None):
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
    self.create_cb = create_cb
    self.generator_cb = generator_cb

  # This function gets executed in two situations:
  #   - In the case of a module function parameterizer, it is called when the
  #   user wants to apply specific parameters to the module. In this case, we
  #   should call the function, wrap the returned module class, and return it.
  #   The result is cached in _MODULE_CACHE.
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
    if cache_key in _MODULE_CACHE:
      return _MODULE_CACHE[cache_key]

    cls = self.func(*args, **kwargs)
    if cls is None:
      raise ValueError("Parameterization function must return module class")

    mod = _module_base(cls,
                       self.extern_name,
                       params=params,
                       create_cb=self.create_cb,
                       generator_cb=self.generator_cb)
    _MODULE_CACHE[cache_key] = mod
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
    return _module_base(to_be_wrapped,
                        extern_name,
                        create_cb=create_msft_module_extern_op)
  return _parameterized_module(to_be_wrapped,
                               extern_name,
                               create_cb=create_msft_module_extern_op)


def _module_base(cls,
                 extern_name: str,
                 create_cb: builtins.function,
                 generator_cb: builtins.function = None,
                 params={}):
  """Wrap a class, making it a PyCDE module."""

  class mod(cls):

    def __init__(self,
                 *args,
                 appid: AppID = None,
                 partition: DesignPartition = None,
                 **kwargs):
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
              value = create_const_zero(type)
          else:
            value = _obj_to_value(input, type)
        else:
          backedge = BackedgeBuilder.current().create(type,
                                                      name,
                                                      mod._pycde_mod.circt_mod,
                                                      loc=loc)
          self.backedges[idx] = backedge
          value = Value(backedge.result)
        inputs[name] = value

      instance_name = cls.__name__
      if "instance_name" in dir(self):
        instance_name = self.instance_name
      instance_name = _BlockContext.current().uniquify_symbol(instance_name)
      # TODO: This is a held Operation*. Add a level of indirection.
      self._instantiation = mod._pycde_mod.instantiate(instance_name,
                                                       inputs,
                                                       appid=appid,
                                                       loc=loc)

      op = self._instantiation.operation
      if partition:
        op.attributes["targetDesignPartition"] = partition._tag

    def output_values(self):
      return {outname: getattr(self, outname) for (outname, _) in mod.outputs()}

    @staticmethod
    def print(out=sys.stdout):
      mod._pycde_mod.print(out)
      print()

    @staticmethod
    def inputs() -> list[(str, mlir.ir.Type)]:
      """Return the list of input ports."""
      return mod._pycde_mod.input_ports

    @staticmethod
    def outputs() -> list[(str, mlir.ir.Type)]:
      """Return the list of input ports."""
      return mod._pycde_mod.output_ports

  mod.__qualname__ = cls.__qualname__
  mod.__name__ = cls.__name__
  mod.__module__ = cls.__module__
  mod._pycde_mod = _SpecializedModule(mod,
                                      params,
                                      extern_name,
                                      create_cb=create_cb,
                                      generator_cb=generator_cb)
  return mod


_current_block_context = ContextVar("current_block_context")


class _BlockContext:
  """Bookkeeping for a scope."""

  def __init__(self):
    self.symbols: set[str] = set()

  @staticmethod
  def current() -> _BlockContext:
    """Get the top-most context in the stack created by `with
    _BlockContext()`."""
    bb = _current_block_context.get(None)
    assert bb is not None
    return bb

  def __enter__(self):
    self._old_system_token = _current_block_context.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_block_context.reset(self._old_system_token)

  def uniquify_symbol(self, sym: str) -> str:
    """Create a unique symbol and add it to the cache. If it is to be preserved,
    the caller must use it as the symbol on a top-level op."""
    ctr = 0
    ret = sym
    while ret in self.symbols:
      ctr += 1
      ret = sym + "_" + str(ctr)
    self.symbols.add(ret)
    return ret


class Generator:
  """
  Represents a generator. Stores the generate function and location of
  the generate call. Generator objects are passed to module-specific generator
  object handlers.
  """

  def __init__(self, gen_func):
    sig = inspect.signature(gen_func)
    if len(sig.parameters) != 1:
      raise ValueError(
          "Generate functions must take one argument and do not support 'self'."
      )
    self.gen_func = gen_func
    self.loc = get_user_loc()


def generator(func):
  """Decorator for generation functions."""
  return Generator(func)


class _GeneratorPortAccess:
  """Get the input ports."""

  __slots__ = ["_mod", "_output_values"]

  def __init__(self, mod: _SpecializedModule):
    self._mod = mod
    self._output_values: dict[str, Value] = {}

  # Support attribute access to block arguments by name
  def __getattr__(self, name):
    if name in self._mod.input_port_lookup:
      idx = self._mod.input_port_lookup[name]
      entry_block = self._mod.circt_mod.regions[0].blocks[0]
      val = entry_block.arguments[idx]
      if name in self._mod.clock_ports:
        return ClockValue(val, ClockType())
      return Value(val)
    if name in self._mod.output_port_lookup:
      if name not in self._output_values:
        raise ValueError("Must set output value before accessing it")
      return self._output_values[name]

    raise AttributeError(f"unknown port name '{name}'")

  def __setattr__(self, name: str, value) -> None:
    if name in _GeneratorPortAccess.__slots__:
      super().__setattr__(name, value)
      return

    if name not in self._mod.output_port_lookup:
      raise ValueError(f"Cannot find output port '{name}'")
    if name in self._output_values:
      raise ValueError(f"Cannot set output '{name}' twice")

    from .behavioral import If
    if If.current() is not None:
      raise ValueError(f"Cannot set output '{name}' inside an if block")

    output_port = self._mod.output_ports[self._mod.output_port_lookup[name]]
    output_port_type = output_port[1]
    if not isinstance(value, Value):
      value = _obj_to_value(value, output_port_type)
    if value.type != output_port_type:
      if value.type == output_port_type.strip:
        value = output_port_type.wrap(value)
      else:
        raise ValueError("Types do not match. Output port type: "
                         f" '{output_port_type}'. Value type: '{value.type}'")
    self._output_values[name] = value

  def set_all_ports(self, port_values: dict[str, Value]):
    """Set all of the output values in a portname -> value dict."""
    for (name, value) in port_values.items():
      self.__setattr__(name, value)


class DesignPartition:
  """Construct a design partition "module" for entities to target."""

  def __init__(self, name: str):
    sym_name = mlir.ir.StringAttr.get(name)
    dp = msft.DesignPartitionOp(sym_name, sym_name)
    parent_mod = dp.operation.parent.attributes["sym_name"]
    # TODO: Add SymbolRefAttr to mlir.ir
    self._tag = mlir.ir.Attribute.parse(f"@{parent_mod}::@{sym_name}")
