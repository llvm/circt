#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .common import (AppID, Clock, Input, InputChannel, Output, OutputChannel,
                     _PyProxy, PortError, Reset)
from .constructs import AssignableSignal, Mux, Wire
from .module import (generator, modparams, Module, ModuleLikeBuilderBase,
                     PortProxyBase)
from .signals import (BitsSignal, BundleSignal, ChannelSignal, ClockSignal,
                      Signal, _FromCirctValue, UIntSignal)
from .support import clog2, optional_dict_to_dict_attr, get_user_loc
from .system import System
from .types import (Any, Bits, Bundle, BundledChannel, Channel,
                    ChannelDirection, ChannelSignaling, StructType, Type,
                    Window, UInt, _FromCirctType)

from .circt import ir
from .circt.dialects import esi as raw_esi, hw, msft

import inspect
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
                                        ir.TypeAttr.get(attr.type._type),
                                        loc=get_user_loc())
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
        inputs=[x.value for x in inputs],
        loc=get_user_loc()).operation.results
    return [_FromCirctValue(x) for x in impl_results]

  def implement_as(self,
                   builtin_type: str,
                   *args: List[Signal],
                   appid: Optional[AppID] = None):
    """Implement this service using a CIRCT implementation named
    'builtin_type'."""
    if appid is None:
      appid = AppID(self.name)
    return self.instantiate_builtin(appid, builtin_type, inputs=args)


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
        raw_esi.RequestConnectionOp(type._type,
                                    self.service_port,
                                    appid._appid,
                                    loc=get_user_loc()).toClient)


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
      raw_esi.ServiceImplClientRecordOp(self.req.relativeAppIDPath,
                                        self.req.servicePort,
                                        ir.TypeAttr.get(self.req.toClient.type),
                                        channelAssignments=channel_assignments,
                                        implDetails=details,
                                        loc=get_user_loc())

  @property
  def client_name(self) -> List[AppID]:
    return [AppID(x) for x in self.req.relativeAppIDPath]

  @property
  def client_name_str(self) -> str:
    return "_".join([str(appid) for appid in self.client_name])

  def assign(self, new_value: BundleSignal):
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
    details = optional_dict_to_dict_attr(details)
    self._rec = raw_esi.ServiceImplRecordOp(
        appID=rec_appid._appid,
        serviceImplName=engine.TypeName,
        implDetails=details,
        isEngine=True,
        loc=get_user_loc(),
    )
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
          loc=get_user_loc(),
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
    # delete the service request op reference.
    bundles.cleanup()

    # Verify the generator did not produce invalid IR.
    sys.mod.operation.verify()

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

  class DeclareRandomAccessMemory(ServiceDecl):
    __name__ = name
    address_width = (depth - 1).bit_length()
    address_type = UInt(address_width)
    write_struct = StructType([('address', address_type), ('data', inner_type)])

    read = Bundle([
        BundledChannel("address", ChannelDirection.FROM, address_type),
        BundledChannel("data", ChannelDirection.TO, inner_type)
    ])
    write = Bundle([
        BundledChannel("req", ChannelDirection.FROM, write_struct),
        BundledChannel("ack", ChannelDirection.TO, Bits(0))
    ])

    def __init__(self):
      super().__init__(self.__class__)
      self.num_autonamed = 0

    def get_write(self, data_type: Type, appid: Optional[AppID] = None):
      """Return a request for a write operation with the given data type.
      Sometimes necessary since the type of an imported RAM may not be
      correct."""
      if appid is None:
        appid = AppID(self.name + "_writer", self.num_autonamed)
      self.num_autonamed += 1
      orig_req = self.write(appid)
      new_arg_type = StructType([
          ("address", UInt(DeclareRandomAccessMemory.address_width)),
          ("data", data_type),
      ])
      xform_bundle_type = Bundle([
          BundledChannel("req", ChannelDirection.FROM, new_arg_type),
          BundledChannel("ack", ChannelDirection.TO, Bits(0))
      ])
      req = orig_req.coerce(xform_bundle_type,
                            from_chan_transform=lambda x: x.bitcast(
                                DeclareRandomAccessMemory.write_struct))
      return req

    @staticmethod
    def _op(sym_name: ir.StringAttr):
      return raw_esi.RandomAccessMemoryDeclOp(
          sym_name,
          ir.TypeAttr.get(inner_type._type),
          ir.IntegerAttr.get(ir.IntegerType.get_signless(64), depth),
          loc=get_user_loc())

  if name is not None:
    DeclareRandomAccessMemory.name = name
    DeclareRandomAccessMemory.__name__ = name
  return DeclareRandomAccessMemory()


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
    return raw_esi.MMIOServiceDeclOp(sym_name, loc=get_user_loc())


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
    write_bundle_type = self.write_req_bundle_type(req.type.inner_type.data)
    bundle = self.write_from_bundle(appid, write_bundle_type)
    resp = bundle.unpack(req=req)['ackTag']
    return resp

  def write_from_bundle(self, appid: AppID,
                        write_bundle_type: Bundle) -> BundleSignal:
    self._materialize_service_decl()

    return cast(
        BundleSignal,
        _FromCirctValue(
            raw_esi.RequestConnectionOp(write_bundle_type._type,
                                        hw.InnerRefAttr.get(
                                            self.symbol,
                                            ir.StringAttr.get("write")),
                                        appid._appid,
                                        loc=get_user_loc()).toClient))

  def read_bundle_type(self, resp_type: Type) -> Bundle:
    """Build a read bundle type for the given data type."""
    resp_type = StructType([
        ("tag", UInt(8)),
        ("data", resp_type),
    ])
    read_bundle_type = Bundle([
        BundledChannel("req", ChannelDirection.FROM, _HostMem.ReadReqType),
        BundledChannel("resp", ChannelDirection.TO, resp_type)
    ])
    return read_bundle_type

  def read_from_bundle(self, appid: AppID,
                       read_bundle_type: Bundle) -> BundleSignal:
    """Request a connection based for a given read bundle type."""
    return cast(
        BundleSignal,
        _FromCirctValue(
            raw_esi.RequestConnectionOp(read_bundle_type._type,
                                        hw.InnerRefAttr.get(
                                            self.symbol,
                                            ir.StringAttr.get("read")),
                                        appid._appid,
                                        loc=get_user_loc()).toClient))

  def read(self, appid: AppID, req: ChannelSignal,
           data_type: Type) -> ChannelSignal:
    """Create a read request to the host memory out of a request channel and
    return the response channel with the specified data type."""

    self._materialize_service_decl()

    read_bundle_type = self.read_bundle_type(data_type)
    bundle_sig = self.read_from_bundle(appid, read_bundle_type)
    resp = bundle_sig.unpack(req=req)['resp']
    return resp

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.HostMemServiceDeclOp(sym_name, loc=get_user_loc())


HostMem = _HostMem()


class _ChannelService(ServiceDecl):
  """Get a single channel connection."""

  def __init__(self):
    super().__init__(self.__class__)

  def from_host(self, name: AppID, type: Type) -> ChannelSignal:
    bundle_type = Bundle(
        [BundledChannel("data", ChannelDirection.TO, Channel(type))])
    self._materialize_service_decl()
    from_host = raw_esi.RequestConnectionOp(bundle_type._type,
                                            hw.InnerRefAttr.get(
                                                self.symbol,
                                                ir.StringAttr.get("from_host")),
                                            name._appid,
                                            loc=get_user_loc())
    from_host = _FromCirctValue(from_host.toClient)
    assert isinstance(from_host, BundleSignal)
    return from_host.unpack()["data"]

  def to_host(self, name: AppID, chan: ChannelSignal) -> None:
    bundle_type = Bundle(
        [BundledChannel("data", ChannelDirection.FROM, chan.type)])
    self._materialize_service_decl()
    to_host = raw_esi.RequestConnectionOp(bundle_type._type,
                                          hw.InnerRefAttr.get(
                                              self.symbol,
                                              ir.StringAttr.get("to_host")),
                                          name._appid,
                                          loc=get_user_loc())
    to_host = _FromCirctValue(to_host.toClient)
    assert isinstance(to_host, BundleSignal)
    to_host.unpack(data=chan)

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.ChannelServiceDeclOp(sym_name, loc=get_user_loc())


ChannelService = _ChannelService()


class _FuncService(ServiceDecl):
  """ESI standard service to request execution of a function."""

  def __init__(self):
    super().__init__(self.__class__)

  def get(self, name: AppID, bundle_type: Bundle) -> BundleSignal:
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
    func_call = raw_esi.RequestConnectionOp(bundle._type,
                                            hw.InnerRefAttr.get(
                                                self.symbol,
                                                ir.StringAttr.get("call")),
                                            name._appid,
                                            loc=get_user_loc())
    to_funcs = _FromCirctValue(func_call.toClient).unpack(result=result)
    return to_funcs['arg']

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.FuncServiceDeclOp(sym_name, loc=get_user_loc())


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
        raw_esi.RequestConnectionOp(func_type._type,
                                    hw.InnerRefAttr.get(
                                        self.symbol, ir.StringAttr.get("call")),
                                    name._appid,
                                    loc=get_user_loc()).toClient)
    assert isinstance(func_call, BundleSignal)
    return func_call

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.CallServiceDeclOp(sym_name, loc=get_user_loc())


CallService = _CallService()


class _Telemetry(ServiceDecl):
  """ESI standard service to report telemetry data."""

  report = Bundle([
      BundledChannel("get", ChannelDirection.TO, Bits(0)),
      BundledChannel("data", ChannelDirection.FROM, Any())
  ])

  def __init__(self):
    super().__init__(self.__class__)

  def report_signal(self, clk: ClockSignal, rst: BitsSignal, name: AppID,
                    data: Signal) -> None:
    """Report a value to the telemetry service. 'data' is the value to report."""
    bundle_type = Bundle([
        BundledChannel("get", ChannelDirection.TO, Bits(0)),
        BundledChannel("data", ChannelDirection.FROM, data.type)
    ])

    report_bundle = self.report(name, bundle_type)
    get_valid_wire = Wire(Bits(1))
    data_channel, data_ready = Channel(data.type).wrap(data, get_valid_wire)
    data_channel = data_channel.buffer(clk, rst, stages=1)
    get_chan = report_bundle.unpack(data=data_channel)['get']
    get_chan = get_chan.buffer(clk, rst, stages=1)
    _, get_valid = get_chan.unwrap(data_ready)
    get_valid_wire.assign(get_valid)

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.TelemetryServiceDeclOp(sym_name, loc=get_user_loc())


Telemetry = _Telemetry()


class TelemetryMMIO(ServiceImplementation):
  """An ESI service implementation which provides telemetry data through an MMIO
  region. Each client request is assigned a register in the MMIO space. When a
  read request is received for the assigned address, it gets routed to the
  assigned client. When a write request is received, it is discarded. The
  assignment table is stored in the manifest."""

  clk = Clock()
  rst = Reset()

  @generator
  def generate(ports, bundles: _ServiceGeneratorBundles) -> bool:
    if len(bundles.to_client_reqs) == 0:
      # No clients to connect to, so we don't need to do anything.
      return True

    mmio_cmd = MMIO.read_write(AppID("__telemetry_mmio"))
    # Assign each telemetry client a register offset in MMIO space.

    offset = 0
    table: Dict[int, AssignableSignal] = {}
    for bundle in bundles.to_client_reqs:
      # Only support 'report' port for telemetry.
      if bundle.port == 'report':
        table[offset] = bundle
        bundle.add_record(details={"offset": offset, "type": "mmio"})
        offset += 8
      else:
        raise ValueError(f"Unrecognized port name: {bundle.port}")

    # Unpack the cmd bundle.
    data_resp_channel = Wire(Channel(MMIODataType), "telemetry_data_resp")
    counted_output = Wire(Channel(MMIODataType), "telemetry_counted_output")
    cmd_channel = mmio_cmd.unpack(data=counted_output)["cmd"]
    counted_output.assign(data_resp_channel)

    # Decode the address to select the client.
    cmd_ready_wire = Wire(Bits(1), "telemetry_cmd_ready")
    cmd, cmd_valid = cmd_channel.unwrap(cmd_ready_wire)
    client_addr_chan, client_addr_ready = Channel(Bits(0)).wrap(
        Bits(0)(0), cmd_valid)
    cmd_ready_wire.assign(client_addr_ready)

    # Build the demux/mux and assign the results of each appropriately.
    read_clients_clog2 = clog2(len(table))
    chan_sel = cmd.offset.as_bits()[3:read_clients_clog2 + 3]
    client_cmd_channels = ChannelDemux(
        sel=chan_sel,
        input=client_addr_chan,
        num_outs=len(table),
        instance_name="telemetry_client_cmd_demux")
    client_data_channels = []
    for (idx, offset) in enumerate(sorted(table.keys())):
      bundle_wire = table[offset]
      bundle_type = bundle_wire.type
      # For telemetry, the client expects a 'get' channel and returns 'data'.
      offset_chan = client_cmd_channels[idx]
      bundle, bundle_froms = bundle_type.pack(get=offset_chan)

      bundle_wire.assign(bundle)
      client_data_channels.append(
          bundle_froms["data"].transform(lambda m: m.as_bits(64)))
    resp_channel = ChannelMux(client_data_channels)
    data_resp_channel.assign(resp_channel)
    return True


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


@modparams
def ListWindowToParallel(serial_window_type: Window):
  """Build a module which converts a 'serial' (bulk-transfer) window of a
  struct-with-a-list into a 'parallel' one. The serial window is expected to
  consist of a single 'header' frame containing the static struct fields plus
  the list `count`, followed by `ceil(count/items_per_frame)` 'data' frames each
  carrying `items_per_frame` list items. The parallel output produces one
  message per list item, replicating the static fields and asserting the `last`
  field on the final item of the list.

  `serial_window_type` must be a Window produced by `Window.serial_of`; the
  `into_type`, `count_bitwidth`, and `items_per_frame` variables are derived
  from it automatically.
  """

  from .types import List as ListType, StructType

  # Derive into_type (the underlying struct) directly from the window.
  into_type = serial_window_type.into

  # The underlying type must be a struct (Window.serial_of wraps non-struct
  # types in a single-field struct, so the .into is always a StructType).
  if not isinstance(into_type, StructType):
    raise ValueError(
        "ListWindowToParallel requires a serial window over a struct type; "
        f"got into-type {into_type}")

  # Collect static and list field names from the underlying struct. There must
  # be exactly one list field.
  static_field_names: List[str] = []
  list_field_name: Optional[str] = None
  for name, ftype in into_type.fields:
    if isinstance(ftype, ListType):
      if list_field_name is not None:
        raise ValueError("ListWindowToParallel requires exactly one list field")
      list_field_name = name
    else:
      static_field_names.append(name)
  if list_field_name is None:
    raise ValueError("ListWindowToParallel requires exactly one list field")
  count_field_name = f"{list_field_name}_count"

  # The serial window must have exactly two frames: a header frame followed
  # by a data frame. The frame names are captured here (rather than hardcoded)
  # so the build logic below indexes the union with the same names the window
  # type was constructed with.
  frames = serial_window_type.frames
  if len(frames) != 2:
    raise ValueError(
        "ListWindowToParallel requires a serial window with exactly 2 frames "
        f"(header followed by data); got {len(frames)}")
  header_frame, data_frame = frames[0], frames[1]
  header_frame_name = header_frame.name
  data_frame_name = data_frame.name
  if header_frame_name is None or data_frame_name is None:
    raise ValueError(
        "ListWindowToParallel requires both serial-window frames to have "
        f"names; got header={header_frame_name!r}, data={data_frame_name!r}")

  # The header frame must contain all of the static struct fields plus a
  # 3-tuple (list_field, 0, count_bitwidth) carrying the bulk-transfer count.
  # Note: Window.frames returns each member as a tuple (name, num_items[,
  # bulk_count_width]); plain strings are also allowed in user-constructed
  # frames, so we accept both forms.
  header_static_names = []
  count_bitwidth: Optional[int] = None
  for member in header_frame.members:
    if isinstance(member, str):
      header_static_names.append(member)
      continue
    if not isinstance(member, tuple):
      raise ValueError(
          f"ListWindowToParallel: unsupported header member {member!r}")
    if len(member) == 3 and member[0] == list_field_name and \
       member[2] is not None and member[2] > 0:
      if count_bitwidth is not None:
        raise ValueError(
            "ListWindowToParallel: header frame has multiple bulk-count "
            f"entries for list field {list_field_name!r}")
      count_bitwidth = member[2]
    elif len(member) >= 2 and \
         (member[1] is None or member[1] == 0) and \
         (len(member) < 3 or not member[2]):
      # Static field encoded as (name, None) or (name, 0).
      header_static_names.append(member[0])
    else:
      raise ValueError(
          "ListWindowToParallel: unexpected member "
          f"{member!r} in header frame; expected static fields plus the "
          f"bulk-count entry for list field {list_field_name!r}")
  if count_bitwidth is None:
    raise ValueError(
        "ListWindowToParallel: header frame is missing the bulk-count entry "
        f"for list field {list_field_name!r}; ensure the serial window was "
        "produced by Window.serial_of")
  if set(header_static_names) != set(static_field_names):
    raise ValueError(
        "ListWindowToParallel: header frame static fields "
        f"{header_static_names} do not match the struct's static fields "
        f"{static_field_names}")

  # The data frame must contain exactly the list field as a 2-tuple
  # (list_field, items_per_frame).
  if len(data_frame.members) != 1:
    raise ValueError(
        "ListWindowToParallel: data frame must contain exactly one member "
        f"(the list field); got {data_frame.members}")
  data_member = data_frame.members[0]
  if not (isinstance(data_member, tuple) and len(data_member) == 2 and
          data_member[0] == list_field_name and data_member[1] is not None):
    raise ValueError(
        "ListWindowToParallel: data frame member must be a 2-tuple "
        f"({list_field_name!r}, items_per_frame); got {data_member!r}")
  items_per_frame = data_member[1]

  if items_per_frame != 1:
    raise NotImplementedError(
        "ListWindowToParallel currently only supports items_per_frame=1, "
        f"got {items_per_frame}")

  parallel_window_type = Window.default_of(into_type)
  parallel_lowered = parallel_window_type.lowered_type

  class ListWindowToParallel(Module):
    """Converts a 'serial' or 'bulk' window into a list to a 'parallel' one."""

    clk = Clock()
    rst = Reset()

    serial_in = InputChannel(serial_window_type)
    parallel_out = OutputChannel(parallel_window_type)

    @generator
    def build(ports):
      from .constructs import Counter
      # State machine for serial-to-parallel conversion. Per the ESI spec, the
      # serial encoding may transmit a list across multiple bursts, each of
      # which is a header (with a non-zero count of items) followed by `count`
      # data frames. The list ends only when a header with count==0 is
      # received -- reaching the end of an individual burst's count does NOT
      # imply the end of the list. To correctly drive the parallel-side
      # `last` field, we therefore peek the next header before emitting the
      # final item of each burst.
      #
      # State encoding (2 bits):
      #   0 (S_WAIT)     - waiting for a header (start of a list, or
      #                    discarding a count==0 terminator with no preceding
      #                    items).
      #   1 (S_EMIT)     - emitting items 1..count-1 of the current burst in
      #                    lockstep with serial_in. The count-th (last) item
      #                    of the burst is consumed silently into a buffer
      #                    register instead of being emitted, so we can
      #                    determine its `last` flag from the next header.
      #   2 (S_PEEK)     - waiting for and consuming the next header so we
      #                    can decide whether the buffered item is the last
      #                    of the entire list (next count == 0) or just the
      #                    last of a burst (next count > 0).
      #   3 (S_EMIT_BUF) - emitting the buffered last-of-burst item with the
      #                    `last` flag set iff the just-peeked header carried
      #                    count==0.
      S_WAIT = Bits(2)(0)
      S_EMIT = Bits(2)(1)
      S_PEEK = Bits(2)(2)
      S_EMIT_BUF = Bits(2)(3)

      state_wire = Wire(Bits(2))
      in_wait = state_wire == S_WAIT
      in_emit = state_wire == S_EMIT
      in_peek = state_wire == S_PEEK
      in_emit_buf = state_wire == S_EMIT_BUF

      # ----- Input handshake -----
      serial_ready_wire = Wire(Bits(1))
      serial_window_sig, serial_valid = ports.serial_in.unwrap(
          serial_ready_wire)
      serial_union = serial_window_sig.unwrap()

      # Header / data variant overlays of the serial union. The valid one is
      # determined by the sender's framing; we only read each variant in the
      # appropriate state. Variant names come from the window type's frame
      # names captured above.
      header_struct = serial_union[header_frame_name]
      data_struct = serial_union[data_frame_name]

      # ----- Counters -----
      zero = UInt(count_bitwidth)(0)
      # `emitted` counts the number of items already emitted in the current
      # burst. It is driven below by a constructs.Counter; we predeclare a
      # Wire so it can be used in handshake/state expressions before the
      # Counter is instantiated.
      emitted_wire = Wire(UInt(count_bitwidth))
      next_emitted = (emitted_wire +
                      UInt(count_bitwidth)(1)).as_uint(count_bitwidth)

      # ----- Header acceptance -----
      # Headers are accepted whenever we are in S_WAIT or S_PEEK and serial_in
      # is valid. The count field is latched on every header accept (we need
      # the peeked terminator's count to compute `out_last`). Static fields
      # are latched only on the *first* header of a list (S_WAIT accepts):
      # per the ESI spec they are constant within a list, but a peeked
      # terminator may carry stale/zero values, so we must not let it
      # overwrite the registers used to construct the buffered item's
      # parallel output.
      hdr_xact = (in_wait | in_peek) & serial_valid
      first_hdr_xact = in_wait & serial_valid

      static_regs: Dict[str, Signal] = {}
      for name in static_field_names:
        static_regs[name] = header_struct[name].reg(ports.clk,
                                                    ports.rst,
                                                    ce=first_hdr_xact,
                                                    name=f"static_{name}")

      new_count = header_struct[count_field_name].as_uint(count_bitwidth)
      count_reg = new_count.reg(ports.clk,
                                ports.rst,
                                ce=hdr_xact,
                                rst_value=0,
                                name="count")
      cur_count = count_reg.as_uint(count_bitwidth)
      # In S_EMIT, the (emitted+1)-th item is the last of the burst.
      is_last_of_burst = next_emitted == cur_count

      # ----- Buffered item -----
      # Latch the last data item of a burst into a register before peeking
      # the next header.
      data_item = data_struct[list_field_name][0]
      consume_burst_last = in_emit & is_last_of_burst & serial_valid
      buf_item = data_item.reg(ports.clk,
                               ports.rst,
                               ce=consume_burst_last,
                               name="buf_item")

      # ----- Parallel output construction -----
      # In S_EMIT we forward the live data_item; in S_EMIT_BUF we emit the
      # buffered item. `last` is only ever set in S_EMIT_BUF, and only when
      # the header just consumed by S_PEEK had count==0.
      out_item = Mux(in_emit_buf, data_item, buf_item)
      out_last = in_emit_buf & (cur_count == zero)

      parallel_fields = {name: static_regs[name] for name in static_field_names}
      parallel_fields[list_field_name] = out_item
      parallel_fields["last"] = out_last
      parallel_struct = parallel_lowered(parallel_fields)
      parallel_window = parallel_window_type.wrap(parallel_struct)

      # parallel_out is valid in two situations:
      # - S_EMIT, with serial_in valid, for non-last-of-burst items (the
      #   last-of-burst item is silently consumed into buf_item instead).
      # - S_EMIT_BUF, where the buffered item is always available.
      out_valid = (in_emit & serial_valid & ~is_last_of_burst) | in_emit_buf

      out_chan, out_ready = Channel(parallel_window_type).wrap(
          parallel_window, out_valid)
      ports.parallel_out = out_chan

      # serial_in.ready:
      # We assert ready in every state EXCEPT S_EMIT_BUF, gated by out_ready.
      # The simpler/more-precise form would split by state (header consumes
      # in S_WAIT/S_PEEK don't strictly need out_ready, and the silent
      # last-of-burst consume in S_EMIT doesn't either), but in practice
      # the consumer's readiness gates overall progress anyway, and
      # collapsing to a single out_ready term shortens the combinational
      # path by removing the is_last_of_burst comparator.
      #
      # The ~in_emit_buf gate is REQUIRED for correctness: in S_EMIT_BUF
      # out_valid is asserted (we are emitting the buffered item), but we
      # are not consuming serial_in. Without this gate, the cycle the
      # buffered item is accepted would also spuriously consume whatever
      # beat is sitting on serial_in (the next list's header/data),
      # dropping it.
      serial_ready_wire.assign(~in_emit_buf & out_ready)

      # ----- Transactions -----
      # TODO: I don't think out_ready is necessary for muxing and looking at it introduces an unnecessary combinational path.
      emit_normal_xact = in_emit & serial_valid & ~is_last_of_burst & out_ready
      emit_buf_xact = in_emit_buf & out_ready

      # ----- emitted counter -----
      # Increment on each non-last emit; clear to 0 when accepting a new
      # header (start of a new burst) or finishing the buffered emit.
      # `increment` and `clear` are mutually exclusive (they require
      # different states), and Counter.clear takes precedence over
      # increment, matching the prior cascaded-Mux semantics.
      reset_emitted = hdr_xact | emit_buf_xact
      emitted_counter = Counter(count_bitwidth)(clk=ports.clk,
                                                rst=ports.rst,
                                                clear=reset_emitted,
                                                increment=emit_normal_xact)
      emitted_wire.assign(emitted_counter.out)

      # ----- State transitions -----
      # The conditions below are mutually exclusive (each requires a specific
      # current state), so the order of the chained Muxes does not matter.
      #   S_WAIT     -> S_EMIT      on hdr_xact & new_count != 0
      #   S_WAIT     -> S_WAIT      on hdr_xact & new_count == 0 (terminator
      #                                                           or empty
      #                                                           list).
      #   S_EMIT     -> S_PEEK      on consume_last
      #   S_PEEK     -> S_EMIT_BUF  on hdr_xact (next header consumed)
      #   S_EMIT_BUF -> S_EMIT      on emit_buf_xact & cur_count != 0
      #   S_EMIT_BUF -> S_WAIT      on emit_buf_xact & cur_count == 0
      new_count_zero = new_count == zero
      cur_count_zero = cur_count == zero

      # Per-state next-state expressions, selected by the current state.
      # Each branch only needs to consider the events relevant to that
      # state, since the multi-input Mux is indexed by state_wire (no
      # need to gate guards with in_<state>). fsm.Machine was considered
      # here, but it is geared toward Moore-style FSMs whose outputs are
      # decoded from named states; this FSM's outputs are derived from
      # the in_<state> wires used pervasively above, so a direct state-
      # indexed Mux is simpler.
      next_in_wait = Mux(hdr_xact & ~new_count_zero, S_WAIT, S_EMIT)
      next_in_emit = Mux(consume_burst_last, S_EMIT, S_PEEK)
      next_in_peek = Mux(hdr_xact, S_PEEK, S_EMIT_BUF)
      # S_EMIT_BUF: stay until emit_buf_xact, then go to S_WAIT if the
      # peeked terminator carried count==0, else back to S_EMIT.
      next_in_emit_buf = Mux(emit_buf_xact, S_EMIT_BUF,
                             Mux(cur_count_zero, S_EMIT, S_WAIT))
      next_state = Mux(state_wire, next_in_wait, next_in_emit, next_in_peek,
                       next_in_emit_buf)

      state_reg = next_state.reg(ports.clk,
                                 ports.rst,
                                 rst_value=0,
                                 name="state")
      state_wire.assign(state_reg)

  return ListWindowToParallel


@modparams
def ListWindowToSerial(parallel_window_type: Window,
                       count_bitwidth: int = 16,
                       items_per_frame: int = 1,
                       fifo_depth: int = 64,
                       meta_fifo_depth: Optional[int] = None):
  """Build a module which converts a 'parallel' window of a struct-with-a-list
  back into a 'serial' (burst-transfer) window. Each parallel input message
  carries one list item plus the static struct fields and a `last` flag marking
  the final item of the list.

  Items are buffered in a data FIFO of `fifo_depth` entries; per-burst
  metadata (static fields, item count, end-of-list flag) is buffered in a
  separate metadata FIFO of `meta_fifo_depth` entries. The output side emits
  one burst transfer per metadata entry: a header frame (carrying the static
  fields and the burst's item count) followed by `count` data frames; if the
  metadata entry is flagged as the end of the list, an additional header
  frame with `count==0` is emitted afterwards as the terminator (per the ESI
  WindowField serial-encoding spec).

  Because metadata is queued (rather than held in a single register), the
  input side does NOT back-pressure the producer once `last` is observed --
  it remains free to accept the next list's items, so multiple lists may be
  in flight at any given time. A burst transfer is closed (and a metadata
  entry pushed) either when the producer asserts `last` (end-of-list) or
  when the data FIFO would otherwise fill (split-on-full back-pressure
  relief). This means the supported list length is unbounded -- not
  constrained by `fifo_depth` -- and lists do not need to wait for each
  other to drain.

  `parallel_window_type` must be a Window produced by `Window.default_of`;
  the `into_type` is derived from it automatically.
  """

  from .types import List as ListType, StructType

  if items_per_frame != 1:
    raise NotImplementedError(
        "ListWindowToSerial currently only supports items_per_frame=1, "
        f"got {items_per_frame}")
  if fifo_depth < 1:
    raise ValueError(f"fifo_depth must be >= 1, got {fifo_depth}")
  if fifo_depth >= (1 << count_bitwidth):
    raise ValueError(
        f"fifo_depth ({fifo_depth}) must fit in count_bitwidth "
        f"({count_bitwidth}) bits since the per-burst count is at most "
        "fifo_depth")
  if meta_fifo_depth is None:
    meta_fifo_depth = max(2, fifo_depth // 4)
  if meta_fifo_depth < 2:
    raise ValueError(f"meta_fifo_depth must be >= 2, got {meta_fifo_depth}")

  # Derive into_type (the underlying struct) directly from the window.
  into_type = parallel_window_type.into

  static_field_names: List[str] = []
  list_field_name: Optional[str] = None
  list_element_type: Optional[Type] = None
  for name, ftype in into_type.fields:
    if isinstance(ftype, ListType):
      if list_field_name is not None:
        raise ValueError("ListWindowToSerial requires exactly one list field")
      list_field_name = name
      list_element_type = ftype.element_type
    else:
      static_field_names.append(name)
  if list_field_name is None:
    raise ValueError("ListWindowToSerial requires exactly one list field")

  serial_window_type = Window.serial_of(into_type, count_bitwidth,
                                        items_per_frame)
  serial_lowered = serial_window_type.lowered_type
  count_field_name = f"{list_field_name}_count"

  # The serial-window lowered type is a union of {header_struct, data_struct}.
  serial_variants = {n: t for (n, t, _) in serial_lowered.fields}
  header_struct_type = serial_variants["header"]
  data_struct_type = serial_variants["data"]

  # The data FIFO carries one list element per slot.
  data_elem_type = list_element_type

  # Burst-count counter width: must hold values 0..fifo_depth.
  bc_bitwidth = max(1, (fifo_depth + 1).bit_length())

  # Metadata FIFO entry: a complete header (static fields + count) plus an
  # is_last flag indicating whether this burst ends the current list (and
  # therefore should be followed by a count==0 terminator footer).
  meta_entry_type = StructType([
      ("hdr", header_struct_type),
      ("is_last", Bits(1)),
  ])

  class ListWindowToSerial(Module):
    """Converts a 'parallel' window of a list into a 'serial' (burst-transfer)
    window using paired data + metadata FIFOs. Multiple lists may be in
    flight simultaneously; lists of arbitrary length are supported by
    splitting them across multiple burst transfers (each with its own
    header), with a final count==0 terminator header per list (per the ESI
    serial-encoding spec).
    """

    clk = Clock()
    rst = Reset()

    parallel_in = InputChannel(parallel_window_type)
    serial_out = OutputChannel(serial_window_type)

    @generator
    def build(ports):
      from .seq import FIFO as SeqFIFO
      from .constructs import ControlReg, Counter

      one_bc = UInt(bc_bitwidth)(1)
      depth_bc = UInt(bc_bitwidth)(fifo_depth)

      # ===== Input side: split into (data, metadata) FIFOs. =====
      par_ready_wire = Wire(Bits(1))
      par_window, par_valid = ports.parallel_in.unwrap(par_ready_wire)
      par_struct = par_window.unwrap()
      par_last = par_struct["last"]
      par_item = par_struct[list_field_name]

      data_fifo = SeqFIFO(data_elem_type, fifo_depth, ports.clk, ports.rst)
      meta_fifo = SeqFIFO(meta_entry_type, meta_fifo_depth, ports.clk,
                          ports.rst)

      # par_ready: require space in BOTH FIFOs unconditionally (a meta push
      # may happen on any par_xact, and computing whether one is needed
      # this cycle would require par_xact -> par_ready -> par_xact comb
      # loop). The slight throughput penalty is acceptable since meta_fifo
      # drains at the bulk-transfer rate.
      par_ready_wire.assign(~data_fifo.full & ~meta_fifo.full)
      par_xact = par_valid & par_ready_wire
      par_xact_last = par_xact & par_last

      data_fifo.push(par_item, par_xact)

      # Track whether we're currently mid-list (i.e. have seen at least one
      # item of the current list but not yet its `last`). `at_list_start`
      # fires on the first par_xact of a new list -- that's where we latch
      # the static fields the spec says should remain constant.
      in_list = ControlReg(ports.clk,
                           ports.rst,
                           asserts=[par_xact & ~par_last],
                           resets=[par_xact_last],
                           name="in_list")
      at_list_start = par_xact & ~in_list

      # Latch static fields at the start of each list. The spec says these
      # fields are constant within a list, but we don't trust the producer
      # blindly: we capture once and then compare for mismatches below.
      latched_static: Dict[str, Signal] = {}
      for name in static_field_names:
        f = par_struct[name]
        latched_static[name] = f.reg(ports.clk,
                                     ports.rst,
                                     ce=at_list_start,
                                     name=f"latched_{name}")

      # Effective static fields used for the meta entry: on at_list_start
      # the register hasn't captured yet (it updates at end-of-cycle), so
      # forward this cycle's value. This matters for single-item lists
      # where at_list_start and par_xact_last coincide.
      def effective_static(name: str) -> Signal:
        return Mux(at_list_start, latched_static[name], par_struct[name])

      # Mismatch detection: on every par_xact except the very first item
      # of a list, any static field that differs from what we latched at
      # the start of the list is a spec violation. Count the cycles in
      # which any field mismatches and report via telemetry.
      mismatch_any = Bits(1)(0)
      for name in static_field_names:
        # `!=` works for any peer-typed Signals.
        diff = par_struct[name] != latched_static[name]
        mismatch_any = mismatch_any | diff
      mismatch_event = par_xact & in_list & mismatch_any

      mismatch_bitwidth = 32
      mismatch_counter = Counter(mismatch_bitwidth)(clk=ports.clk,
                                                    rst=ports.rst,
                                                    clear=Bits(1)(0),
                                                    increment=mismatch_event)
      Telemetry.report_signal(
          ports.clk,
          ports.rst,
          AppID("listStaticFieldMismatches"),
          mismatch_counter.out,
      )

      # Per-burst item counter: counts items pushed into data_fifo so far
      # for the current (in-progress) burst. burst_count_wire is the
      # registered value BEFORE this cycle's push, so the count of items
      # *including* this cycle's push is burst_count_wire + 1.
      burst_count_wire = Wire(UInt(bc_bitwidth))
      next_burst_count = (burst_count_wire + one_bc).as_uint(bc_bitwidth)

      # Push a meta entry whenever a burst boundary occurs:
      #   - par_xact_last: end-of-list (is_last=1).
      #   - drain_split:   non-last xact that fills the data FIFO. Without
      #                    this, a long list with no `last` would stall
      #                    once data_fifo fills. is_last=0.
      # The two are mutually exclusive (drain_split requires ~par_last).
      drain_split = par_xact & ~par_last & (next_burst_count == depth_bc)
      meta_push_xact = par_xact_last | drain_split

      # Build the meta entry: count snapshot plus the static fields we
      # latched at the start of the current list (rather than trusting
      # whatever the producer happens to be presenting at the boundary
      # cycle).
      hdr_fields: Dict[str, Signal] = {
          name: effective_static(name) for name in static_field_names
      }
      hdr_fields[count_field_name] = next_burst_count.as_bits(count_bitwidth)
      hdr_value = header_struct_type(hdr_fields)
      meta_entry_value = meta_entry_type({
          "hdr": hdr_value,
          "is_last": par_xact_last,
      })
      meta_fifo.push(meta_entry_value, meta_push_xact)

      # Burst counter: increment on each par_xact, clear on meta_push_xact.
      # Counter.clear takes precedence over .increment, so when both fire
      # in the same cycle the counter resets to 0 (correct: that cycle's
      # item is the last of the burst, and the next burst starts at 0).
      burst_counter = Counter(bc_bitwidth)(clk=ports.clk,
                                           rst=ports.rst,
                                           clear=meta_push_xact,
                                           increment=par_xact)
      burst_count_wire.assign(burst_counter.out)

      # ===== Output side: drain meta + data FIFOs into serial frames. =====
      # State (2 bits):
      #   S_IDLE   - waiting for a meta entry; pop one to start a burst.
      #   S_HDR    - emit the latched header.
      #   S_DATA   - emit `count` data frames.
      #   S_FOOTER - emit a count==0 terminator (only if cur_is_last).
      S_IDLE = Bits(2)(0)
      S_HDR = Bits(2)(1)
      S_DATA = Bits(2)(2)
      S_FOOTER = Bits(2)(3)

      state_wire = Wire(Bits(2))
      in_idle = state_wire == S_IDLE
      in_hdr = state_wire == S_HDR
      in_data = state_wire == S_DATA
      in_footer = state_wire == S_FOOTER

      # Pop one meta entry when we're idle and meta is available; latch it
      # into cur_meta so it's stable through HDR/DATA/FOOTER states.
      meta_pop_rden = Wire(Bits(1))
      meta_value = meta_fifo.pop(meta_pop_rden)
      meta_avail = ~meta_fifo.empty
      arm_burst = in_idle & meta_avail
      meta_pop_rden.assign(arm_burst)

      cur_meta = meta_value.reg(ports.clk,
                                ports.rst,
                                ce=arm_burst,
                                name="cur_meta")
      cur_hdr = cur_meta["hdr"]
      cur_count = cur_hdr[count_field_name].as_uint(count_bitwidth)
      cur_is_last = cur_meta["is_last"]

      # Footer struct: same static fields as the latched header, count=0.
      footer_fields: Dict[str, Signal] = {
          name: cur_hdr[name] for name in static_field_names
      }
      footer_fields[count_field_name] = Bits(count_bitwidth)(0)
      footer_value = header_struct_type(footer_fields)

      # Per-burst emitted-item counter, cleared at burst end.
      emitted_wire = Wire(UInt(count_bitwidth))
      next_emitted = (emitted_wire +
                      UInt(count_bitwidth)(1)).as_uint(count_bitwidth)
      last_item_in_burst = next_emitted == cur_count

      # Data FIFO pop.
      data_pop_rden = Wire(Bits(1))
      data_value = data_fifo.pop(data_pop_rden)
      data_valid = ~data_fifo.empty

      # Output union variants.
      header_union = serial_lowered(("header", cur_hdr))
      footer_union = serial_lowered(("header", footer_value))
      data_struct_value = data_struct_type({list_field_name: [data_value]})
      data_union = serial_lowered(("data", data_struct_value))

      # Mux the union by current state. The S_IDLE slot is don't-care
      # since out_valid is deasserted there; we plug in data_union to
      # avoid an extra distinct constant.
      out_union = Mux(state_wire, data_union, header_union, data_union,
                      footer_union)
      out_window = serial_window_type.wrap(out_union)

      # out_valid:
      #   - S_HDR / S_FOOTER: always (driven from registers).
      #   - S_DATA:           when the next item is queued in data_fifo.
      out_valid = in_hdr | in_footer | (in_data & data_valid)
      out_chan, out_ready = Channel(serial_window_type).wrap(
          out_window, out_valid)
      ports.serial_out = out_chan

      # Transactions.
      hdr_xact = in_hdr & out_ready
      data_xact = in_data & data_valid & out_ready
      burst_done = data_xact & last_item_in_burst
      footer_xact = in_footer & out_ready
      data_pop_rden.assign(data_xact)

      # Emitted-counter: clear at burst_done, increment on every other
      # data xact. Counter.clear takes precedence over .increment.
      emitted_counter = Counter(count_bitwidth)(clk=ports.clk,
                                                rst=ports.rst,
                                                clear=burst_done,
                                                increment=data_xact &
                                                ~last_item_in_burst)
      emitted_wire.assign(emitted_counter.out)

      # Per-state next-state expressions, selected by the current state.
      # Transitions:
      #   S_IDLE   -> S_HDR    on arm_burst
      #   S_HDR    -> S_DATA   on hdr_xact (cur_count is always >= 1, since
      #                        meta is only pushed on a real boundary that
      #                        includes at least one item).
      #   S_DATA   -> S_FOOTER on burst_done & cur_is_last
      #   S_DATA   -> S_IDLE   on burst_done & ~cur_is_last
      #   S_FOOTER -> S_IDLE   on footer_xact
      next_in_idle = Mux(arm_burst, S_IDLE, S_HDR)
      next_in_hdr = Mux(hdr_xact, S_HDR, S_DATA)
      next_in_data = Mux(burst_done, S_DATA, Mux(cur_is_last, S_IDLE, S_FOOTER))
      next_in_footer = Mux(footer_xact, S_FOOTER, S_IDLE)
      next_state = Mux(state_wire, next_in_idle, next_in_hdr, next_in_data,
                       next_in_footer)
      state_reg = next_state.reg(ports.clk,
                                 ports.rst,
                                 rst_value=0,
                                 name="state")
      state_wire.assign(state_reg)

  return ListWindowToSerial
