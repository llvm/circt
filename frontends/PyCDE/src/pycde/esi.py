#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pycde.system import System
from .module import (Generator, _module_base, _BlockContext,
                     _GeneratorPortAccess, _SpecializedModule)
from pycde.value import ChannelValue, ClockValue, PyCDEValue, Value
from .common import AppID, Input, Output, InputChannel, OutputChannel, _PyProxy
from circt.dialects import esi as raw_esi, hw, msft
from circt.support import BackedgeBuilder
from pycde.pycde_types import ChannelType, ClockType, PyCDEType, types

import mlir.ir as ir

from pathlib import Path
import shutil
from typing import List, Optional, Type

__dir__ = Path(__file__).parent

ToServer = InputChannel
FromServer = OutputChannel


class ToFromServer:
  """A bidirectional channel declaration."""

  def __init__(self, to_server_type: Type, to_client_type: Type):
    self.to_server_type = ChannelType(raw_esi.ChannelType.get(to_server_type))
    self.to_client_type = ChannelType(raw_esi.ChannelType.get(to_client_type))


class ServiceDecl(_PyProxy):
  """Declare an ESI service interface."""

  def __init__(self, cls: Type):
    self.name = cls.__name__
    if hasattr(cls, "_op"):
      self._op = cls._op
    else:
      self._op = raw_esi.CustomServiceDeclOp
    for (attr_name, attr) in vars(cls).items():
      if isinstance(attr, InputChannel):
        setattr(self, attr_name,
                _RequestToServerConn(self, attr.type, None, attr_name))
      elif isinstance(attr, OutputChannel):
        setattr(self, attr_name,
                _RequestToClientConn(self, None, attr.type, attr_name))
      elif isinstance(attr, ToFromServer):
        setattr(
            self, attr_name,
            _RequestToFromServerConn(self, attr.to_server_type,
                                     attr.to_client_type, attr_name))
      elif isinstance(attr, (Input, Output)):
        raise TypeError(
            "Input and Output are not allowed in ESI service declarations. " +
            " Use InputChannel and OutputChannel instead.")

  def _materialize_service_decl(self) -> str:
    """Create the ServiceDeclOp. We must do this lazily since this class gets
    instantiated when the code is read, rather than during `System` generation
    time. Return its symbol name."""

    from .system import System, _OpCache
    curr_sys: System = System.current()
    op_cache: _OpCache = curr_sys._op_cache
    sym_name = op_cache.get_pyproxy_symbol(self)
    if sym_name is None:
      sym_name, install = op_cache.create_symbol(self)
      self.symbol = ir.StringAttr.get(sym_name)
      with curr_sys._get_ip():
        decl = self._op(self.symbol)
        install(decl)

      if self._op is raw_esi.CustomServiceDeclOp:
        ports_block = ir.Block.create_at_start(decl.ports, [])
        with ir.InsertionPoint.at_block_begin(ports_block):
          for (_, attr) in self.__dict__.items():
            if isinstance(attr, _RequestToServerConn):
              raw_esi.ToServerOp(attr._name,
                                 ir.TypeAttr.get(attr.to_server_type))
            elif isinstance(attr, _RequestToClientConn):
              raw_esi.ToClientOp(attr._name,
                                 ir.TypeAttr.get(attr.to_client_type))
            elif isinstance(attr, _RequestToFromServerConn):
              raw_esi.ServiceDeclInOutOp(attr._name,
                                         ir.TypeAttr.get(attr.to_server_type),
                                         ir.TypeAttr.get(attr.to_client_type))
    return sym_name

  def instantiate_builtin(self,
                          builtin: str,
                          result_types: List[PyCDEType] = [],
                          inputs: List[PyCDEValue] = []):
    """Implement a service using an implementation builtin to CIRCT. Needs the
    input ports which the implementation expects and returns the outputs."""

    # TODO: figure out a way to verify the ports during this call.
    impl_results = raw_esi.ServiceInstanceOp(
        result=result_types,
        service_symbol=ir.FlatSymbolRefAttr.get(
            self._materialize_service_decl()),
        impl_type=ir.StringAttr.get(builtin),
        inputs=[x.value for x in inputs]).operation.results
    return [Value(x) for x in impl_results]


class _RequestConnection:
  """Parent to 'request' proxy classes. Constructed as attributes on the
  ServiceDecl class. Provides syntactic sugar for constructing service
  connection requests."""

  def __init__(self, decl: ServiceDecl, to_server_type: Optional[ir.Type],
               to_client_type: Optional[ir.Type], attr_name: str):
    self.decl = decl
    self._name = ir.StringAttr.get(attr_name)
    self.to_server_type = ChannelType(
        to_server_type) if to_server_type is not None else None
    self.to_client_type = ChannelType(
        to_client_type) if to_client_type is not None else None

  @property
  def service_port(self) -> hw.InnerRefAttr:
    return hw.InnerRefAttr.get(self.decl.symbol, self._name)


class _RequestToServerConn(_RequestConnection):

  def __call__(self, chan: ChannelValue, chan_name: str = ""):
    self.decl._materialize_service_decl()
    raw_esi.RequestToServerConnectionOp(
        self.service_port, chan.value,
        ir.ArrayAttr.get([ir.StringAttr.get(chan_name)]))


class _RequestToClientConn(_RequestConnection):

  def __call__(self, chan_name: str = "", type: Optional[PyCDEType] = None):
    self.decl._materialize_service_decl()
    if type is None:
      type = self.to_client_type
      if type == types.any:
        raise ValueError(
            "If service port has type 'any', then 'type' must be specified.")
    if not isinstance(type, ChannelType):
      type = types.channel(type)
    req_op = raw_esi.RequestToClientConnectionOp(
        type, self.service_port,
        ir.ArrayAttr.get([ir.StringAttr.get(chan_name)]))
    return ChannelValue(req_op)


class _RequestToFromServerConn(_RequestConnection):

  def __call__(self,
               to_server_channel: ChannelValue,
               chan_name: str = "",
               to_client_type: Optional[PyCDEType] = None):
    self.decl._materialize_service_decl()
    type = to_client_type
    if type is None:
      type = self.to_client_type
      if type == types.any:
        raise ValueError(
            "If service port has type 'any', then 'type' must be specified.")
    if not isinstance(type, ChannelType):
      type = types.channel(type)
    to_client = raw_esi.RequestInOutChannelOp(
        self.to_client_type, self.service_port, to_server_channel.value,
        ir.ArrayAttr.get([ir.StringAttr.get(chan_name)]))
    return ChannelValue(to_client)


def Cosim(decl: ServiceDecl, clk, rst):
  """Implement a service via cosimulation."""
  decl.instantiate_builtin("cosim", [], [clk, rst])


def CosimBSP(user_module):
  """Wrap and return a cosimulation 'board support package' containing
  'user_module'"""
  from .module import module, generator
  from .common import Clock, Input

  @module
  class top:
    clk = Clock()
    rst = Input(types.int(1))

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)
      raw_esi.ServiceInstanceOp(result=[],
                                service_symbol=None,
                                impl_type=ir.StringAttr.get("cosim"),
                                inputs=[ports.clk.value, ports.rst.value])

      System.current().add_packaging_step(top.package)

    @staticmethod
    def package(sys: System):
      """Run the packaging to create a cosim package."""
      # TODO: this only works in-tree. Make it work in packages as well.
      build_dir = __dir__.parents[4]
      bin_dir = build_dir / "bin"
      lib_dir = build_dir / "lib"
      circt_inc_dir = build_dir / "tools" / "circt" / "include" / "circt"
      assert circt_inc_dir.exists(), "Only works in the CIRCT build directory"
      esi_inc_dir = circt_inc_dir / "Dialect" / "ESI"
      hw_src = sys.hw_output_dir
      shutil.copy(lib_dir / "libEsiCosimDpiServer.so", hw_src)
      shutil.copy(bin_dir / "driver.cpp", hw_src)
      shutil.copy(bin_dir / "driver.sv", hw_src)
      shutil.copy(esi_inc_dir / "ESIPrimitives.sv", hw_src)
      shutil.copy(esi_inc_dir / "Cosim_DpiPkg.sv", hw_src)
      shutil.copy(esi_inc_dir / "Cosim_Endpoint.sv", hw_src)
      shutil.copy(__dir__ / "Makefile.cosim", sys.output_directory)
      shutil.copy(sys.hw_output_dir / "schema.capnp", sys.runtime_output_dir)

  return top


class NamedChannelValue(ChannelValue):
  """A ChannelValue with the name of the client request."""

  def __init__(self, input_chan: ir.Value, client_name: List[str]):
    self.client_name = client_name
    super().__init__(input_chan)


class _OutputChannelSetter:
  """Return a list of these as a proxy for a 'request to client connection'.
  Users should call the 'assign' method with the `ChannelValue` which they
  have implemented for this request."""

  def __init__(self, req: raw_esi.RequestToClientConnectionOp,
               old_chan_to_replace: ChannelValue):
    self.type = ChannelType(req.toClient.type)
    self.client_name = req.clientNamePath
    self._chan_to_replace = old_chan_to_replace

  def assign(self, new_value: ChannelValue):
    """Assign the generated channel to this request."""
    if self._chan_to_replace is None:
      name_str = ".".join(self.client_name)
      raise ValueError(f"{name_str} has already been connected.")
    if new_value.type != self.type:
      raise TypeError(
          f"ChannelType mismatch. Expected {self.type}, got {new_value.type}.")
    msft.replaceAllUsesWith(self._chan_to_replace, new_value.value)
    self._chan_to_replace = None


class _ServiceGeneratorChannels:
  """Provide access to the channels which the service generator is responsible
  for connecting up."""

  def __init__(self, mod: _SpecializedModule,
               req: raw_esi.ServiceImplementReqOp):
    self._req = req
    portReqsBlock = req.portReqs.blocks[0]

    # Find the input channel requests and store named versions of the values.
    self._input_reqs = [
        NamedChannelValue(x.toServer, x.clientNamePath)
        for x in portReqsBlock
        if isinstance(x, raw_esi.RequestToServerConnectionOp)
    ]

    # Find the output channel requests and store the settable proxies.
    num_output_ports = len(mod.output_port_lookup)
    self._output_reqs = [
        _OutputChannelSetter(req, self._req.results[num_output_ports + idx])
        for idx, req in enumerate(portReqsBlock)
        if isinstance(req, raw_esi.RequestToClientConnectionOp)
    ]
    assert len(self._output_reqs) == len(req.results) - num_output_ports

  @property
  def to_server_reqs(self) -> List[NamedChannelValue]:
    """Get the list of incoming channels from the 'to server' connection
    requests."""
    return self._input_reqs

  @property
  def to_client_reqs(self) -> List[_OutputChannelSetter]:
    return self._output_reqs

  def check_unconnected_outputs(self):
    for req in self._output_reqs:
      if req._chan_to_replace is not None:
        name_str = ".".join(req.client_name)
        raise ValueError(f"{name_str} has not been connected.")


def ServiceImplementation(decl: Optional[ServiceDecl]):
  """A generator for a service implementation. Must contain a @generator method
  which will be called whenever required to implement the server. Said generator
  function will be called with the same 'ports' argument as modules and a
  'channels' argument containing lists of the input and output channels which
  need to be connected to the service being implemented."""

  def wrap(service_impl, decl: Optional[ServiceDecl] = decl):

    def instantiate_cb(mod: _SpecializedModule, instance_name: str,
                       inputs: dict, appid: AppID, loc):
      # Each instantiation of the ServiceImplementation has its own
      # registration.
      opts = _service_generator_registry.register(mod)
      decl_sym = None
      if decl is not None:
        decl_sym = ir.FlatSymbolRefAttr.get(decl._materialize_service_decl())
      return raw_esi.ServiceInstanceOp(
          result=[t for _, t in mod.output_ports],
          service_symbol=decl_sym,
          impl_type=_ServiceGeneratorRegistry._impl_type_name,
          inputs=[inputs[pn].value for pn, _ in mod.input_ports],
          impl_opts=opts,
          loc=loc)

    def generate(generator: Generator, spec_mod: _SpecializedModule,
                 serviceReq: raw_esi.ServiceInstanceOp):
      arguments = serviceReq.operation.operands
      with ir.InsertionPoint(
          serviceReq), generator.loc, BackedgeBuilder(), _BlockContext():
        # Insert generated code after the service instance op.
        ports = _GeneratorPortAccess(spec_mod, arguments)

        # Enter clock block implicitly if only one clock given
        clk = None
        if len(spec_mod.clock_ports) == 1:
          clk_port = list(spec_mod.clock_ports.values())[0]
          clk = ClockValue(arguments[clk_port], ClockType())
          clk.__enter__()

        # Run the generator.
        channels = _ServiceGeneratorChannels(spec_mod, serviceReq)
        rc = generator.gen_func(ports, channels=channels)
        if rc is None:
          rc = True
        elif not isinstance(rc, bool):
          raise ValueError("Generators must a return a bool or None")
        ports.check_unconnected_outputs()
        channels.check_unconnected_outputs()

        # Replace the output values from the service implement request op with
        # the generated values. Erase the service implement request op.
        for port_name, port_value in ports._output_values.items():
          port_num = spec_mod.output_port_lookup[port_name]
          msft.replaceAllUsesWith(serviceReq.operation.results[port_num],
                                  port_value.value)
        serviceReq.operation.erase()

        if clk is not None:
          clk.__exit__(None, None, None)

        return rc

    return _module_base(service_impl,
                        extern_name=None,
                        generator_cb=generate,
                        instantiate_cb=instantiate_cb)

  return wrap


class _ServiceGeneratorRegistry:
  """Class to register individual service instance generators. Should be a
  singleton."""
  _registered = False
  _impl_type_name = ir.StringAttr.get("pycde")

  def __init__(self):
    self._registry = {}

    # Register myself with ESI so I can dispatch to my internal registry.
    assert _ServiceGeneratorRegistry._registered is False, \
      "Cannot instantiate more than one _ServiceGeneratorRegistry"
    raw_esi.registerServiceGenerator(
        _ServiceGeneratorRegistry._impl_type_name.value,
        self._implement_service)
    _ServiceGeneratorRegistry._registered = True

  def register(self, service_implementation: _SpecializedModule) -> ir.DictAttr:
    """Register a ServiceImplementation generator with the PyCDE generator.
    Called when the ServiceImplamentation is defined."""

    # Create unique name for the service instance.
    basename = service_implementation.name
    name = basename
    ctr = 0
    while name in self._registry:
      ctr += 1
      name = basename + "_" + str(ctr)
    name_attr = ir.StringAttr.get(name)
    self._registry[name_attr] = (service_implementation, System.current())
    return ir.DictAttr.get({"name": name_attr})

  def _implement_service(self, req: ir.Operation):
    """This is the callback which the ESI connect-services pass calls. Dispatch
    to the op-specified generator."""
    assert isinstance(req.opview, raw_esi.ServiceImplementReqOp)
    opts = ir.DictAttr(req.attributes["impl_opts"])
    impl_name = opts["name"]
    if impl_name not in self._registry:
      return False
    (impl, sys) = self._registry[impl_name]
    with sys:
      return impl.generate(serviceReq=req.opview)


_service_generator_registry = _ServiceGeneratorRegistry()


def DeclareRandomAccessMemory(inner_type: PyCDEType,
                              depth: int,
                              name: Optional[str] = None):
  """Declare an ESI RAM with elements of type 'inner_type' and has 'depth' of
  them. Memories (as with all ESI services) are not actually instantiated until
  the place where you specify the implementation."""

  @ServiceDecl
  class DeclareRandomAccessMemory:
    __name__ = name
    address_type = types.int((depth - 1).bit_length())
    write_type = types.struct([('address', address_type), ('data', inner_type)])

    read = ToFromServer(to_server_type=address_type, to_client_type=inner_type)
    write = ToFromServer(to_server_type=write_type, to_client_type=types.i0)

    @staticmethod
    def _op(sym_name: ir.StringAttr):
      return raw_esi.RandomAccessMemoryDeclOp(
          sym_name, ir.TypeAttr.get(inner_type),
          ir.IntegerAttr.get(ir.IntegerType.get_signless(64), depth))

  if name is not None:
    DeclareRandomAccessMemory.name = name
    DeclareRandomAccessMemory.__name__ = name
  return DeclareRandomAccessMemory


def _import_ram_decl(sys: "System", ram_op: raw_esi.RandomAccessMemoryDeclOp):
  """Create a DeclareRandomAccessMemory object from an existing CIRCT op and
  install it in the sym cache."""
  from .system import _OpCache
  ram = DeclareRandomAccessMemory(inner_type=PyCDEType(ram_op.innerType.value),
                                  depth=ram_op.depth.value,
                                  name=ram_op.sym_name.value)
  cache: _OpCache = sys._op_cache
  sym, install = cache.create_symbol(ram)
  assert sym == ram_op.sym_name.value, "don't support imported module renames"
  ram.symbol = ir.StringAttr.get(sym)
  install(ram_op)
  return ram
