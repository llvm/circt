from pycde.system import System
from .module import (Generator, _module_base, _BlockContext,
                     _GeneratorPortAccess, _SpecializedModule)
from pycde.value import ChannelValue, ClockValue
from .common import AppID, Input, Output, InputChannel, OutputChannel, _PyProxy
from circt.dialects import esi as raw_esi, hw, msft
from circt.support import BackedgeBuilder
from pycde.pycde_types import ChannelType, ClockType, PyCDEType, types

import mlir.ir as ir

from typing import List, Optional, Type

ToServer = InputChannel
FromServer = OutputChannel


class ToFromServer:
  """A bidirectional channel declaration."""

  def __init__(self, to_server_type: Type, to_client_type: Type):
    self.to_server_type = raw_esi.ChannelType.get(to_server_type)
    self.to_client_type = raw_esi.ChannelType.get(to_client_type)


class ServiceDecl(_PyProxy):
  """Declare an ESI service interface."""

  def __init__(self, cls: Type):
    self.name = cls.__name__
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
      with curr_sys._get_ip():
        decl = raw_esi.ServiceDeclOp(ir.StringAttr.get(sym_name))
        install(decl)
      ports_block = ir.Block.create_at_start(decl.ports, [])
      with ir.InsertionPoint.at_block_begin(ports_block):
        for (_, attr) in self.__dict__.items():
          if isinstance(attr, _RequestToServerConn):
            raw_esi.ToServerOp(attr._name, ir.TypeAttr.get(attr.to_server_type))
          elif isinstance(attr, _RequestToClientConn):
            raw_esi.ToClientOp(attr._name, ir.TypeAttr.get(attr.to_client_type))
          elif isinstance(attr, _RequestToFromServerConn):
            raw_esi.ServiceDeclInOutOp(attr._name,
                                       ir.TypeAttr.get(attr.to_server_type),
                                       ir.TypeAttr.get(attr.to_client_type))
      self.symbol = ir.StringAttr.get(sym_name)
    return sym_name


class _RequestConnection:
  """Parent to 'request' proxy classes. Constructed as attributes on the
  ServiceDecl class. Provides syntactic sugar for constructing service
  connection requests."""

  def __init__(self, decl: ServiceDecl, to_server_type: Optional[PyCDEType],
               to_client_type: Optional[PyCDEType], attr_name: str):
    self.decl = decl
    self._name = ir.StringAttr.get(attr_name)
    self.to_server_type = to_server_type
    self.to_client_type = to_client_type

  @property
  def service_port(self) -> hw.InnerRefAttr:
    return hw.InnerRefAttr.get(self.decl.symbol, self._name)


class _RequestToServerConn(_RequestConnection):

  def __call__(self, chan_name: str, chan: ChannelValue):
    self.decl._materialize_service_decl()
    raw_esi.RequestToServerConnectionOp(
        self.service_port, chan.value,
        ir.ArrayAttr.get([ir.StringAttr.get(chan_name)]))


class _RequestToClientConn(_RequestConnection):

  def __call__(self, chan_name: str, type: Optional[PyCDEType] = None):
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
               chan_name: str,
               to_server_channel: ChannelValue,
               to_client_type: Optional[PyCDEType] = None):
    self.decl._materialize_service_decl()
    if to_client_type is None:
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

  # TODO: better modeling and implementation capacity. The below is just
  # temporary.
  raw_esi.ServiceInstanceOp(result=[],
                            service_symbol=ir.FlatSymbolRefAttr.get(
                                decl._materialize_service_decl()),
                            impl_type=ir.StringAttr.get("cosim"),
                            inputs=[clk.value, rst.value])


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
    self.type = ChannelType(req.receiving.type)
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
    input_req_ops = [
        x for x in portReqsBlock
        if isinstance(x, raw_esi.RequestToServerConnectionOp)
    ]
    start_inputs_chan_num = len(mod.input_port_lookup)
    assert len(input_req_ops) == len(req.inputs) - len(mod.input_port_lookup)
    self._input_reqs = [
        NamedChannelValue(input_value, req.clientNamePath)
        for input_value, req in zip(
            list(req.inputs)[start_inputs_chan_num:], input_req_ops)
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


def ServiceImplementation(decl: ServiceDecl):
  """A generator for a service implementation. Must contain a @generator method
  which will be called whenever required to implement the server. Said generator
  function will be called with the same 'ports' argument as modules and a
  'channels' argument containing lists of the input and output channels which
  need to be connected to the service being implemented."""

  def wrap(service_impl, decl: ServiceDecl = decl):

    def instantiate_cb(mod: _SpecializedModule, instance_name: str,
                       inputs: dict, appid: AppID, loc):
      # Each instantiation of the ServiceImplementation has its own
      # registration.
      opts = _service_generator_registry.register(mod)
      return raw_esi.ServiceInstanceOp(
          result=[t for _, t in mod.output_ports],
          service_symbol=ir.FlatSymbolRefAttr.get(
              decl._materialize_service_decl()),
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
