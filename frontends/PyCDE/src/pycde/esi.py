#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .common import (AppID, Clock, Input, InputChannel, Output, OutputChannel,
                     _PyProxy, PortError, Reset)
from .constructs import AssignableSignal, Mux, Wire
from .module import (generator, modparams, Module, ModuleLikeBuilderBase,
                     PortProxyBase)
from .signals import (BitsSignal, BundleSignal, ChannelSignal, Signal,
                      _FromCirctValue, UIntSignal)
from .support import clog2, optional_dict_to_dict_attr, get_user_loc
from .system import System
from .types import (Any, Bits, Bundle, BundledChannel, Channel,
                    ChannelDirection, StructType, Type, UInt, types,
                    _FromCirctType)

from .circt import ir
from .circt.dialects import esi as raw_esi, hw, msft

from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import typing

import atexit

__dir__ = Path(__file__).parent


@atexit.register
def _cleanup():
  raw_esi.cleanup()


FlattenStructPorts = "esi.portFlattenStructs"
PortInSuffix = "esi.portInSuffix"
PortOutSuffix = "esi.portOutSuffix"
PortValidSuffix = "esi.portValidSuffix"
PortReadySuffix = "esi.portReadySuffix"
PortRdenSuffix = "esi.portRdenSuffix"
PortEmptySuffix = "esi.portEmptySuffix"


class ServiceDecl(_PyProxy):
  """Declare an ESI service interface."""

  def __init__(self, cls: type):
    self.name = cls.__name__
    if hasattr(cls, "_op"):
      self._op = cls._op
    else:
      self._op = raw_esi.CustomServiceDeclOp
    for (attr_name, attr) in vars(cls).items():
      if isinstance(attr, Bundle):
        setattr(self, attr_name, _RequestConnection(self, attr, attr_name))
      elif isinstance(attr, (Input, Output)):
        raise TypeError(
            "Input and Output are not allowed in ESI service declarations."
            " Use Bundles instead.")

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
            if isinstance(attr, _RequestConnection):
              raw_esi.ServiceDeclPortOp(attr._name,
                                        ir.TypeAttr.get(attr.type._type))
    return sym_name

  def instantiate_builtin(self,
                          appid: AppID,
                          builtin: str,
                          result_types: List[Type] = [],
                          inputs: List[Signal] = []):
    """Implement a service using an implementation builtin to CIRCT. Needs the
    input ports which the implementation expects and returns the outputs."""

    # TODO: figure out a way to verify the ports during this call.
    impl_results = raw_esi.ServiceInstanceOp(
        result=result_types,
        appID=appid._appid,
        service_symbol=ir.FlatSymbolRefAttr.get(
            self._materialize_service_decl()),
        impl_type=ir.StringAttr.get(builtin),
        inputs=[x.value for x in inputs]).operation.results
    return [_FromCirctValue(x) for x in impl_results]


class _RequestConnection:
  """Indicates a service with a 'from server' port. Call to create a 'from
  server' (to client) connection request."""

  def __init__(self, decl: ServiceDecl, type: Bundle, attr_name: str):
    self.decl = decl
    self._name = ir.StringAttr.get(attr_name)
    self.type = type

  @property
  def service_port(self) -> hw.InnerRefAttr:
    return hw.InnerRefAttr.get(self.decl.symbol, self._name)

  def __call__(self, appid: AppID, type: Optional[Bundle] = None):
    if type is None:
      type = self.type
    self.decl._materialize_service_decl()
    return _FromCirctValue(
        raw_esi.RequestConnectionOp(type._type, self.service_port,
                                    appid._appid).toClient)


def Cosim(decl: ServiceDecl, clk, rst):
  """Implement a service via cosimulation."""
  decl.instantiate_builtin(AppID("cosim", 0), "cosim", [], [clk, rst])


class EngineModule(Module):
  """A module which implements an ESI engines. Engines have the responsibility
  of transporting messages between two different devices."""

  @property
  def TypeName(self):
    assert False, "Engine modules must have a TypeName property."

  def __init__(self, appid: AppID, **inputs):
    super(EngineModule, self).__init__(appid=appid, **inputs)

  @property
  def appid(self) -> AppID:
    return AppID(self.inst.attributes["esi.appid"])


class NamedChannelValue(ChannelSignal):
  """A ChannelValue with the name of the client request."""

  def __init__(self, input_chan: ir.Value, client_name: List[str]):
    self.client_name = client_name
    super().__init__(input_chan, _FromCirctType(input_chan.type))


class _OutputBundleSetter(AssignableSignal):
  """Return a list of these as a proxy for a 'request to client connection'.
  Users should call the 'assign' method with the `ChannelValue` which they
  have implemented for this request."""

  def __init__(self, req: raw_esi.ServiceImplementConnReqOp,
               rec: raw_esi.ServiceImplRecordOp,
               old_value_to_replace: ir.OpResult):
    self.req = req
    self.rec = rec
    self.type: Bundle = _FromCirctType(req.toClient.type)
    self.port = hw.InnerRefAttr(req.servicePort).name.value
    self._bundle_to_replace: Optional[ir.OpResult] = old_value_to_replace

  def add_record(self,
                 channel_assignments: Optional[Dict] = None,
                 details: Optional[Dict[str, object]] = None):
    """Add a record to the manifest for this client request. Generally used to
    give the runtime necessary information about how to connect to the client
    through the generated service. For instance, offsets into an MMIO space."""

    channel_assignments = optional_dict_to_dict_attr(channel_assignments)
    details = optional_dict_to_dict_attr(details)

    with get_user_loc(), ir.InsertionPoint.at_block_begin(
        self.rec.reqDetails.blocks[0]):
      raw_esi.ServiceImplClientRecordOp(
          self.req.relativeAppIDPath,
          self.req.servicePort,
          ir.TypeAttr.get(self.req.toClient.type),
          channelAssignments=channel_assignments,
          implDetails=details,
      )

  @property
  def client_name(self) -> List[AppID]:
    return [AppID(x) for x in self.req.relativeAppIDPath]

  @property
  def client_name_str(self) -> str:
    return "_".join([str(appid) for appid in self.client_name])

  def assign(self, new_value: ChannelSignal):
    """Assign the generated channel to this request."""
    if self._bundle_to_replace is None:
      name_str = ".".join(self.client_name)
      raise ValueError(f"{name_str} has already been connected.")
    if new_value.type != self.type:
      raise TypeError(
          f"Channel type mismatch. Expected {self.type}, got {new_value.type}.")
    msft.replaceAllUsesWith(self._bundle_to_replace, new_value.value)
    self._bundle_to_replace = None

  def cleanup(self):
    """Null out all the references to all the ops to allow them to be GC'd."""
    self.req = None
    self.rec = None


class EngineServiceRecord:
  """Represents a record in the engine section of the manifest."""

  def __init__(self,
               engine: EngineModule,
               details: Optional[Dict[str, object]] = None):
    rec_appid = AppID(f"{engine.appid.name}_record", engine.appid.index)
    self._rec = raw_esi.ServiceImplRecordOp(appID=rec_appid._appid,
                                            serviceImplName=engine.TypeName,
                                            implDetails=details,
                                            isEngine=True)
    self._rec.regions[0].blocks.append()

  def add_record(self,
                 client: _OutputBundleSetter,
                 channel_assignments: Optional[Dict] = None,
                 details: Optional[Dict[str, object]] = None):
    """Add a record to the manifest for this client request. Generally used to
    give the runtime necessary information about how to connect to the client
    through the generated service. For instance, offsets into an MMIO space."""

    channel_assignments = optional_dict_to_dict_attr(channel_assignments)
    details = optional_dict_to_dict_attr(details)

    with get_user_loc(), ir.InsertionPoint.at_block_begin(
        self._rec.reqDetails.blocks[0]):
      raw_esi.ServiceImplClientRecordOp(
          client.req.relativeAppIDPath,
          client.req.servicePort,
          ir.TypeAttr.get(client.req.toClient.type),
          channelAssignments=channel_assignments,
          implDetails=details,
      )


class _ServiceGeneratorBundles:
  """Provide access to the bundles which the service generator is responsible
  for connecting up."""

  def __init__(self, mod: ModuleLikeBuilderBase,
               req: raw_esi.ServiceImplementReqOp,
               rec: raw_esi.ServiceImplRecordOp):
    self._req = req
    self._rec = rec
    portReqsBlock = req.portReqs.blocks[0]

    # Find the output channel requests and store the settable proxies.
    num_output_ports = len(mod.outputs)
    to_client_reqs = [
        req for req in portReqsBlock
        if isinstance(req, raw_esi.ServiceImplementConnReqOp)
    ]
    self._output_reqs = [
        _OutputBundleSetter(req, rec, self._req.results[num_output_ports + idx])
        for idx, req in enumerate(to_client_reqs)
    ]
    assert len(self._output_reqs) == len(req.results) - num_output_ports

  def emit_engine(self,
                  engine: EngineModule,
                  details: Dict[str, object] = None):
    """Emit and return an engine record."""
    details = optional_dict_to_dict_attr(details)
    with get_user_loc(), ir.InsertionPoint(self._rec):
      return EngineServiceRecord(engine, details)

  @property
  def to_client_reqs(self) -> List[_OutputBundleSetter]:
    return self._output_reqs

  def check_unconnected_outputs(self):
    for req in self._output_reqs:
      if req._bundle_to_replace is not None:
        name_str = str(req.client_name)
        raise ValueError(f"{name_str} has not been connected.")

  def cleanup(self):
    """Null out all the references to all the ops to allow them to be GC'd."""
    for req in self._output_reqs:
      req.cleanup()
    self._req = None
    self._rec = None


class ServiceImplementationModuleBuilder(ModuleLikeBuilderBase):
  """Define how to build ESI service implementations. Unlike Modules, there is
  no distinction between definition and instance -- ESI service providers are
  built where they are instantiated."""

  def instantiate(self, impl, inputs: Dict[str, Signal], appid: AppID):
    # Each instantiation of the ServiceImplementation has its own
    # registration.
    opts = _service_generator_registry.register(impl)

    # Create the op.
    decl_sym = None
    if impl.decl is not None:
      decl_sym = ir.FlatSymbolRefAttr.get(impl.decl._materialize_service_decl())
    return raw_esi.ServiceInstanceOp(
        result=[p.type._type for p in self.outputs],
        appID=appid._appid,
        service_symbol=decl_sym,
        impl_type=_ServiceGeneratorRegistry._impl_type_name,
        inputs=[inputs[p.name].value for p in self.inputs],
        impl_opts=opts,
        loc=self.loc)

  def generate_svc_impl(self, sys: System,
                        serviceReq: raw_esi.ServiceImplementReqOp,
                        record_op: raw_esi.ServiceImplRecordOp) -> bool:
    """"Generate the service inline and replace the `ServiceInstanceOp` which is
    being implemented."""

    assert len(self.generators) == 1
    generator: Generator = list(self.generators.values())[0]
    ports = self.generator_port_proxy(serviceReq.operation.operands, self)
    with sys, self.GeneratorCtxt(self, ports, serviceReq, generator.loc):

      # Run the generator.
      bundles = _ServiceGeneratorBundles(self, serviceReq, record_op)
      rc = generator.gen_func(ports, bundles=bundles)
      if rc is None:
        rc = True
      elif not isinstance(rc, bool):
        raise ValueError("Generators must a return a bool or None")
      ports._check_unconnected_outputs()
      bundles.check_unconnected_outputs()

      # Replace the output values from the service implement request op with
      # the generated values. Erase the service implement request op.
      for idx, port_value in enumerate(ports._output_values):
        msft.replaceAllUsesWith(serviceReq.operation.results[idx],
                                port_value.value)

    # Erase the service request op so as to avoid bundles with no consumers.
    serviceReq.operation.erase()

    # The service implementation generator could have instantiated new modules,
    # so we need to generate them. Don't run the appID indexer since during a
    # pass, the IR can be invalid and the indexers assumes it is valid.
    sys.generate(skip_appid_index=True)
    # Now that the bundles should be assigned, we can cleanup the bundles and
    # delete the service request op.
    bundles.cleanup()

    return rc


class ServiceImplementation(Module):
  """A generator for a service implementation. Must contain a @generator method
  which will be called whenever required to implement the server. Said generator
  function will be called with the same 'ports' argument as modules and a
  'channels' argument containing lists of the input and output channels which
  need to be connected to the service being implemented."""

  BuilderType = ServiceImplementationModuleBuilder

  def __init__(self, decl: Optional[ServiceDecl], **kwargs):
    """Instantiate a service provider for service declaration 'decl'. If decl,
    implementation is expected to handle any and all service declarations."""

    self.decl = decl
    super().__init__(**kwargs)

  @property
  def name(self):
    return self.__class__.__name__


class _ServiceGeneratorRegistry:
  """Class to register individual service instance generators. Should be a
  singleton."""
  _registered = False
  _impl_type_name = ir.StringAttr.get("pycde")

  def __init__(self) -> None:
    self._registry: Dict[ir.StringAttr, Tuple[ServiceImplementation,
                                              System]] = {}

    # Register myself with ESI so I can dispatch to my internal registry.
    assert _ServiceGeneratorRegistry._registered is False, \
      "Cannot instantiate more than one _ServiceGeneratorRegistry"
    raw_esi.registerServiceGenerator(
        _ServiceGeneratorRegistry._impl_type_name.value,
        self._implement_service)
    _ServiceGeneratorRegistry._registered = True

  def register(self,
               service_implementation: ServiceImplementation) -> ir.DictAttr:
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

  def _implement_service(self, req: ir.Operation, decl: ir.Operation,
                         rec: ir.Operation):
    """This is the callback which the ESI connect-services pass calls. Dispatch
    to the op-specified generator."""
    assert isinstance(req.opview, raw_esi.ServiceImplementReqOp)
    opts = ir.DictAttr(req.attributes["impl_opts"])
    impl_name = opts["name"]
    if impl_name not in self._registry:
      return False
    (impl, sys) = self._registry[impl_name]
    return impl._builder.generate_svc_impl(sys,
                                           serviceReq=req.opview,
                                           record_op=rec.opview)


_service_generator_registry = _ServiceGeneratorRegistry()


def DeclareRandomAccessMemory(inner_type: Type,
                              depth: int,
                              name: Optional[str] = None):
  """Declare an ESI RAM with elements of type 'inner_type' and has 'depth' of
  them. Memories (as with all ESI services) are not actually instantiated until
  the place where you specify the implementation."""

  @ServiceDecl
  class DeclareRandomAccessMemory:
    __name__ = name
    address_type = types.int((depth - 1).bit_length())
    write_struct = types.struct([('address', address_type),
                                 ('data', inner_type)])

    read = Bundle([
        BundledChannel("address", ChannelDirection.FROM, address_type),
        BundledChannel("data", ChannelDirection.TO, inner_type)
    ])
    write = Bundle([
        BundledChannel("req", ChannelDirection.FROM, write_struct),
        BundledChannel("ack", ChannelDirection.TO, Bits(0))
    ])

    @staticmethod
    def _op(sym_name: ir.StringAttr):
      return raw_esi.RandomAccessMemoryDeclOp(
          sym_name, ir.TypeAttr.get(inner_type._type),
          ir.IntegerAttr.get(ir.IntegerType.get_signless(64), depth))

  if name is not None:
    DeclareRandomAccessMemory.name = name
    DeclareRandomAccessMemory.__name__ = name
  return DeclareRandomAccessMemory


def _import_ram_decl(sys: "System", ram_op: raw_esi.RandomAccessMemoryDeclOp):
  """Create a DeclareRandomAccessMemory object from an existing CIRCT op and
  install it in the sym cache."""
  from .system import _OpCache
  ram = DeclareRandomAccessMemory(inner_type=Type(ram_op.innerType.value),
                                  depth=ram_op.depth.value,
                                  name=ram_op.sym_name.value)
  cache: _OpCache = sys._op_cache
  sym, install = cache.create_symbol(ram)
  assert sym == ram_op.sym_name.value, "don't support imported module renames"
  ram.symbol = ir.StringAttr.get(sym)
  install(ram_op)
  return ram


class PureModuleBuilder(ModuleLikeBuilderBase):
  """Defines how an ESI `PureModule` gets built."""

  @property
  def circt_mod(self):
    from .system import System
    sys: System = System.current()
    ret = sys._op_cache.get_circt_mod(self)
    if ret is None:
      return sys._create_circt_mod(self)
    return ret

  def create_op(self, sys: System, symbol):
    """Callback for creating a ESIPureModule op."""
    mod = raw_esi.ESIPureModuleOp(symbol, loc=self.loc, ip=sys._get_ip())
    for k, v in self.attributes.items():
      mod.attributes[k] = v
    return mod

  def scan_cls(self):
    """Scan the class for input/output ports and generators. (Most `ModuleLike`
    will use these.) Store the results for later use."""

    super().scan_cls()

    if len(self.inputs) != 0 or len(self.outputs) != 0 or len(self.clocks) != 0:
      raise PortError("ESI pure modules cannot have ports")

  def create_port_proxy(self):
    """Since pure ESI modules don't have any ports, this function is pretty
    boring."""
    proxy_attrs = {}
    return type(self.modcls.__name__ + "Ports", (PortProxyBase,), proxy_attrs)

  def add_external_port_accessors(self):
    """Since we don't have ports, do nothing."""
    pass

  def generate(self):
    """Fill in (generate) this module. Only supports a single generator
    currently."""
    if len(self.generators) != 1:
      raise ValueError("Must have exactly one generator.")
    g: Generator = list(self.generators.values())[0]

    entry_block = self.circt_mod.add_entry_block()
    ports = self.generator_port_proxy(None, self)
    with self.GeneratorCtxt(self, ports, entry_block, g.loc):
      g.gen_func(ports)


class PureModule(Module):
  """A pure ESI module has no ports and contains only instances of modules with
  only ESI ports and connections between said instances. Use ESI services for
  external communication."""

  BuilderType = PureModuleBuilder

  @staticmethod
  def input_port(name: str, type: Type):
    from .dialects import esi
    return esi.ESIPureModuleInputOp(type, name)

  @staticmethod
  def output_port(name: str, signal: Signal):
    from .dialects import esi
    return esi.ESIPureModuleOutputOp(name, signal)

  @staticmethod
  def param(name: str, type: Type = None):
    """Create a parameter in the resulting module."""
    from .dialects import esi
    from .circt import ir
    if type is None:
      type_attr = ir.TypeAttr.get(ir.NoneType.get())
    else:
      type_attr = ir.TypeAttr.get(type._type)
    esi.ESIPureModuleParamOp(name, type_attr)


MMIODataType = Bits(64)
MMIOReadWriteCmdType = StructType([
    ("write", Bits(1)),
    ("offset", UInt(32)),
    ("data", MMIODataType),
])


@ServiceDecl
class MMIO:
  """ESI standard service to request access to an MMIO region.

  For now, each client request gets a 1KB region of memory."""

  read = Bundle([
      BundledChannel("offset", ChannelDirection.TO, UInt(32)),
      BundledChannel("data", ChannelDirection.FROM, MMIODataType)
  ])

  read_write = Bundle([
      BundledChannel("cmd", ChannelDirection.TO, MMIOReadWriteCmdType),
      BundledChannel("data", ChannelDirection.FROM, MMIODataType)
  ])

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.MMIOServiceDeclOp(sym_name)


class _HostMem(ServiceDecl):
  """ESI standard service to request read or write access to host memory."""

  TagType = UInt(8)

  ReadReqType = StructType([
      ("address", UInt(64)),
      ("tag", TagType),
  ])

  def __init__(self):
    super().__init__(self.__class__)

  def write_req_bundle_type(self, data_type: Type) -> Bundle:
    """Build a write request bundle type for the given data type."""
    write_req_type = StructType([
        ("address", UInt(64)),
        ("tag", UInt(8)),
        ("data", data_type),
    ])
    return Bundle([
        BundledChannel("req", ChannelDirection.FROM, write_req_type),
        BundledChannel("ackTag", ChannelDirection.TO, _HostMem.TagType),
    ])

  def write_req_channel_type(self, data_type: Type) -> StructType:
    """Return a write request struct type for 'data_type'."""
    return StructType([
        ("address", UInt(64)),
        ("tag", _HostMem.TagType),
        ("data", data_type),
    ])

  def wrap_write_req(self, address: UIntSignal, data: Type, tag: UIntSignal,
                     valid: BitsSignal) -> Tuple[ChannelSignal, BitsSignal]:
    """Create the proper channel type for a write request and use it to wrap the
    given request arguments. Returns the Channel signal and a ready bit."""
    inner_type = self.write_req_channel_type(data.type)
    return Channel(inner_type).wrap(
        inner_type({
            "address": address,
            "tag": tag,
            "data": data,
        }), valid)

  def write(self, appid: AppID, req: ChannelSignal) -> ChannelSignal:
    """Create a write request to the host memory out of a request channel."""
    # Extract the data type from the request channel and call the helper to get
    # the write bundle type for the req channel.
    req_data_type = req.type.inner_type.data
    write_bundle_type = self.write_req_bundle_type(req_data_type)

    bundle = cast(
        BundleSignal,
        _FromCirctValue(
            raw_esi.RequestConnectionOp(
                write_bundle_type._type,
                hw.InnerRefAttr.get(self.symbol, ir.StringAttr.get("write")),
                appid._appid).toClient))

  # Create a read request to the host memory out of a request channel and return
  # the response channel with the specified data type.
  def read(self, appid: AppID, req: ChannelSignal,
           data_type: Type) -> ChannelSignal:
    self._materialize_service_decl()

    resp_type = StructType([
        ("tag", UInt(8)),
        ("data", data_type),
    ])
    read_bundle_type = Bundle([
        BundledChannel("req", ChannelDirection.FROM, _HostMem.ReadReqType),
        BundledChannel("resp", ChannelDirection.TO, resp_type)
    ])

    bundle = cast(
        BundleSignal,
        _FromCirctValue(
            raw_esi.RequestConnectionOp(
                read_bundle_type._type,
                hw.InnerRefAttr.get(self.symbol, ir.StringAttr.get("read")),
                appid._appid).toClient))
    resp = bundle.unpack(req=req)['resp']
    return resp

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.HostMemServiceDeclOp(sym_name)


HostMem = _HostMem()


@ServiceDecl
class _ChannelServiceDecl:
  """Get a single channel connection."""

  from_host = Bundle([BundledChannel("data", ChannelDirection.TO, Any())])
  to_host = Bundle([BundledChannel("data", ChannelDirection.FROM, Any())])


class _ChannelService:

  def from_host(self, name: AppID, type: Type) -> ChannelSignal:
    bundle_type = Bundle(
        [BundledChannel("data", ChannelDirection.TO, Channel(type))])
    from_host_bundle = _ChannelServiceDecl.from_host(name, bundle_type)
    assert isinstance(from_host_bundle, BundleSignal)
    return from_host_bundle.unpack()["data"]

  def to_host(self, name: AppID, chan: ChannelSignal) -> None:
    bundle_type = Bundle(
        [BundledChannel("data", ChannelDirection.FROM, chan.type)])
    to_host_bundle = _ChannelServiceDecl.to_host(name, bundle_type)
    assert isinstance(to_host_bundle, BundleSignal)
    return to_host_bundle.unpack(data=chan)


ChannelService = _ChannelService()


class _FuncService(ServiceDecl):
  """ESI standard service to request execution of a function."""

  def __init__(self):
    super().__init__(self.__class__)

  def get_coerced(self, name: AppID, bundle_type: Bundle) -> BundleSignal:
    """Treat any bi-directional bundle as a function by getting a proper
    function bundle with the appropriate types, then renaming the channels to
    match the 'bundle_type'. Returns a bundle signal of type 'bundle_type'."""

    from .constructs import Wire
    bundle_channels = bundle_type.channels
    if len(bundle_channels) != 2:
      raise ValueError("Bundle must have exactly two channels.")

    # Find the FROM and TO channels.
    to_channel_bc: Optional[BundledChannel] = None
    from_channel_bc: Optional[BundledChannel] = None
    if bundle_channels[0].direction == ChannelDirection.TO:
      to_channel_bc = bundle_channels[0]
    else:
      from_channel_bc = bundle_channels[0]
    if bundle_channels[1].direction == ChannelDirection.TO:
      to_channel_bc = bundle_channels[1]
    else:
      from_channel_bc = bundle_channels[1]
    if to_channel_bc is None or from_channel_bc is None:
      raise ValueError("Bundle must have one channel in each direction.")

    # Get the function channels and wire them up to create the non-function
    # bundle 'bundle_type'.
    from_channel = Wire(from_channel_bc.channel)
    arg_channel = self.get_call_chans(name, to_channel_bc.channel, from_channel)
    ret_bundle, from_chans = bundle_type.pack(
        **{to_channel_bc.name: arg_channel})
    from_channel.assign(from_chans[from_channel_bc.name])
    return ret_bundle

  def get(self, name: AppID, func_type: Bundle) -> BundleSignal:
    """Expose a bundle to the host as a function. Bundle _must_ have 'arg' and
    'result' channels going FROM the server and TO the server, respectively."""
    self._materialize_service_decl()

    func_call = _FromCirctValue(
        raw_esi.RequestConnectionOp(
            func_type._type,
            hw.InnerRefAttr.get(self.symbol, ir.StringAttr.get("call")),
            name._appid).toClient)
    assert isinstance(func_call, BundleSignal)
    return func_call

  def get_call_chans(self, name: AppID, arg_type: Type,
                     result: Signal) -> ChannelSignal:
    """Expose a function to the ESI system. Arguments:
      'name' is an AppID which is the function name.
      'arg_type' is the type of the argument to the function.
      'result' is a Signal which is the result of the function. Typically, it'll
      be a Wire which gets assigned to later on.

      Returns a Signal of 'arg_type' type which is the argument value from the
      caller."""

    bundle = Bundle([
        BundledChannel("arg", ChannelDirection.TO, arg_type),
        BundledChannel("result", ChannelDirection.FROM, result.type)
    ])
    self._materialize_service_decl()
    func_call = raw_esi.RequestConnectionOp(
        bundle._type, hw.InnerRefAttr.get(self.symbol,
                                          ir.StringAttr.get("call")),
        name._appid)
    to_funcs = _FromCirctValue(func_call.toClient).unpack(result=result)
    return to_funcs['arg']

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.FuncServiceDeclOp(sym_name)


FuncService = _FuncService()


class _CallService(ServiceDecl):
  """ESI standard service to request execution of a function."""

  def __init__(self):
    super().__init__(self.__class__)

  def call(self, name: AppID, arg: ChannelSignal,
           result_type: Type) -> ChannelSignal:
    """Call a function with the given argument. 'arg' must be a ChannelSignal
    with the argument value."""
    func_bundle = Bundle([
        BundledChannel("arg", ChannelDirection.FROM, arg.type),
        BundledChannel("result", ChannelDirection.TO, result_type)
    ])
    call_bundle = self.get(name, func_bundle)
    bundle_rets = call_bundle.unpack(arg=arg)
    return bundle_rets['result']

  def get(self, name: AppID, func_type: Bundle) -> BundleSignal:
    """Expose a bundle to the host as a function. Bundle _must_ have 'arg' and
    'result' channels going FROM the server and TO the server, respectively."""
    self._materialize_service_decl()

    func_call = _FromCirctValue(
        raw_esi.RequestConnectionOp(
            func_type._type,
            hw.InnerRefAttr.get(self.symbol, ir.StringAttr.get("call")),
            name._appid).toClient)
    assert isinstance(func_call, BundleSignal)
    return func_call

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.CallServiceDeclOp(sym_name)


CallService = _CallService()


def package(sys: System):
  """Package all ESI collateral."""

  import shutil
  shutil.copy(__dir__ / "ESIPrimitives.sv", sys.hw_output_dir)


def TaggedDemux(num_clients: int,
                channel_type: Channel) -> typing.Type["TaggedDemuxImpl"]:
  """Construct a tagged demultiplexer for a given tagged data type.
  'tagged_data_type' is assumed to be a struct with a 'tag' field and a 'data'
  field OR a UInt representing the tag itself. Demux the data to the appropriate
  output channel based on the tag."""

  class TaggedDemuxImpl(Module):
    clk = Clock()
    rst = Reset()

    in_ = Input(channel_type)

    output_names = [f"out{i}" for i in range(num_clients)]
    for idx in range(num_clients):
      locals()[output_names[idx]] = Output(channel_type)

    def get_out(self, idx: int) -> ChannelSignal:
      return getattr(self, self.output_names[idx])

    @generator
    def build(ports) -> None:
      upstream_ready_wire = Wire(Bits(1))
      upstream_data, upstream_valid = ports.in_.unwrap(upstream_ready_wire)
      upstream_data_type = upstream_data.type

      upstream_ready = Bits(1)(1)
      for idx in range(num_clients):
        if isinstance(upstream_data_type, StructType):
          tag = upstream_data.tag
        elif isinstance(upstream_data_type, UInt):
          tag = upstream_data
        else:
          raise TypeError("TaggedDemux input must be a struct or UInt.")
        output_valid = upstream_valid & (tag == UInt(8)(idx))
        output_ch, output_ready = channel_type.wrap(upstream_data, output_valid)
        setattr(ports, TaggedDemuxImpl.output_names[idx], output_ch)
        upstream_ready = upstream_ready & output_ready

      upstream_ready_wire.assign(upstream_ready)

  return TaggedDemuxImpl


def ChannelDemux2(data_type: Type):

  class ChannelDemux2(Module):
    """Combinational 2-way channel demultiplexer for valid/ready signaling."""

    sel = Input(Bits(1))
    inp = Input(Channel(data_type))
    output0 = Output(Channel(data_type))
    output1 = Output(Channel(data_type))

    @generator
    def generate(ports) -> None:
      input_ready = Wire(Bits(1))
      input, input_valid = ports.inp.unwrap(input_ready)

      output0 = input
      output0_valid = input_valid & (ports.sel == Bits(1)(0))
      output0_ch, output0_ready = Channel(data_type).wrap(
          output0, output0_valid)
      ports.output0 = output0_ch

      output1 = input
      output1_valid = input_valid & (ports.sel == Bits(1)(1))
      output1_ch, output1_ready = Channel(data_type).wrap(
          output1, output1_valid)
      ports.output1 = output1_ch

      input_ready.assign((output0_ready & (ports.sel == Bits(1)(0))) |
                         (output1_ready & (ports.sel == Bits(1)(1))))

  return ChannelDemux2


def ChannelDemux(input: ChannelSignal,
                 sel: BitsSignal,
                 num_outs: int,
                 instance_name: Optional[str] = None) -> List[ChannelSignal]:
  """Build a demultiplexer of ESI channels. 'num_outs' is the number of outputs,
  sel it the select signal, and input is the input channel. Function simply
  passes though to the module and is provided to legacy reasons."""
  demux = ChannelDemuxMod(input.type, num_outs)(input=input,
                                                sel=sel,
                                                instance_name=instance_name)
  return [getattr(demux, f"output_{i}") for i in range(num_outs)]


@modparams
def ChannelDemuxMod(input_channel_type: Type, num_outs: int):
  """Build a demultiplexer of ESI channels."""

  class ChannelDemuxImpl(Module):
    input = Input(input_channel_type)
    sel = Input(Bits(clog2(num_outs)))

    # Add an output port for each read client.
    for i in range(num_outs):
      locals()[f"output_{i}"] = Output(input_channel_type)

    @generator
    def build(ports) -> None:
      dmux2 = ChannelDemux2(ports.input.type)

      def build_tree(inter_input: ChannelSignal, inter_sel: BitsSignal,
                     inter_num_outs: int, path: str) -> List[ChannelSignal]:
        """Builds a binary tree of demuxes to demux the input channel."""
        if inter_num_outs == 0:
          return []
        if inter_num_outs == 1:
          return [inter_input]

        demux2 = dmux2(sel=inter_sel[-1].as_bits(),
                       inp=inter_input,
                       instance_name=f"demux2_path{path}")
        next_sel = inter_sel[:-1].as_bits()
        tree0 = build_tree(demux2.output0, next_sel, (inter_num_outs + 1) // 2,
                           path + "0")
        tree1 = build_tree(demux2.output1, next_sel, (inter_num_outs + 1) // 2,
                           path + "1")
        return tree0 + tree1

      outputs = build_tree(ports.input, ports.sel, num_outs, "")
      for idx, output in enumerate(outputs):
        setattr(ports, f"output_{idx}", output)

  return ChannelDemuxImpl


def ChannelMux2(data_type: Channel):

  class ChannelMux2(Module):
    """2 channel arbiter with priority given to input0. Valid/ready only.
    Combinational."""
    # TODO: implement some fairness.

    input0 = Input(data_type)
    input1 = Input(data_type)
    output_channel = Output(data_type)

    @generator
    def generate(ports):
      input0_ready = Wire(Bits(1))
      input0_data, input0_valid = ports.input0.unwrap(input0_ready)
      input1_ready = Wire(Bits(1))
      input1_data, input1_valid = ports.input1.unwrap(input1_ready)

      output_idx = ~input0_valid
      data_mux = Mux(output_idx, input0_data, input1_data)
      valid_mux = Mux(output_idx, input0_valid, input1_valid)
      output_channel, output_ready = data_type.wrap(data_mux, valid_mux)
      ports.output_channel = output_channel

      input0_ready.assign(output_ready & ~output_idx)
      input1_ready.assign(output_ready & output_idx)

  return ChannelMux2


def ChannelMux(input_channels: List[ChannelSignal]) -> ChannelSignal:
  """Build a channel multiplexer of ESI channels. Ideally, this would be a
  parameterized module with an array of output channels, but the current ESI
  channel-port lowering doesn't deal with arrays of channels. Independent of the
  signaling protocol."""

  assert len(input_channels) > 0
  mux2 = ChannelMux2(input_channels[0].type)

  def build_tree(inter_input_channels: List[ChannelSignal]) -> ChannelSignal:
    assert len(inter_input_channels) > 0
    if len(inter_input_channels) == 1:
      return inter_input_channels[0]
    if len(inter_input_channels) == 2:
      m = mux2(input0=inter_input_channels[0], input1=inter_input_channels[1])
      return m.output_channel
    m0_out = build_tree(inter_input_channels[:len(inter_input_channels) // 2])
    m1_out = build_tree(inter_input_channels[len(inter_input_channels) // 2:])
    m = mux2(input0=m0_out, input1=m1_out)
    return m.output_channel

  return build_tree(input_channels)


@modparams
def Mailbox(type):
  """Constructs a module which stores an ESI message until it is read. Acts as a
  sink -- always accepts new messages, dropping the current unread one if
  necessary. It also allows snooping on the message contents."""
  if isinstance(type, Channel):
    type = type.inner_type

  class Mailbox(Module):
    clk = Clock()
    rst = Input(Bits(1))

    input = InputChannel(type)
    output = OutputChannel(type)

    data = Output(type)
    valid = Output(Bits(1))

    @generator
    def generate(ports):
      input_ready = Bits(1)(1)
      input_data, input_valid = ports.input.unwrap(input_ready)
      input_xact = input_valid & input_ready
      valid_reset = Wire(Bits(1))
      data = input_data.reg(ports.clk, ports.rst, ce=input_xact)
      valid = input_valid.reg(ports.clk, ports.rst, ce=input_xact | valid_reset)

      output, output_ready = Channel(type).wrap(data, valid)
      valid_reset.assign(output_ready & valid)
      ports.output = output
      ports.data = data
      ports.valid = valid

  return Mailbox
