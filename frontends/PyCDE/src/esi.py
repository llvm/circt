#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .common import (AppID, Input, Output, _PyProxy, PortError)
from .module import Generator, Module, ModuleLikeBuilderBase, PortProxyBase
from .signals import BundleSignal, ChannelSignal, Signal, Struct, _FromCirctValue
from .system import System
from .types import (Any, Bits, Bundle, BundledChannel, Channel,
                    ChannelDirection, StructType, Type, types, UInt,
                    _FromCirctType)

from .circt import ir
from .circt.dialects import esi as raw_esi, hw, msft

from pathlib import Path
from typing import Dict, List, Optional, Tuple

__dir__ = Path(__file__).parent

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


class NamedChannelValue(ChannelSignal):
  """A ChannelValue with the name of the client request."""

  def __init__(self, input_chan: ir.Value, client_name: List[str]):
    self.client_name = client_name
    super().__init__(input_chan, _FromCirctType(input_chan.type))


class _OutputBundleSetter:
  """Return a list of these as a proxy for a 'request to client connection'.
  Users should call the 'assign' method with the `ChannelValue` which they
  have implemented for this request."""

  def __init__(self, req: raw_esi.ServiceImplementConnReqOp,
               old_value_to_replace: ir.OpResult):
    self.type: Bundle = _FromCirctType(req.toClient.type)
    self.client_name = req.relativeAppIDPath
    self._bundle_to_replace: Optional[ir.OpResult] = old_value_to_replace

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


class _ServiceGeneratorBundles:
  """Provide access to the bundles which the service generator is responsible
  for connecting up."""

  def __init__(self, mod: ModuleLikeBuilderBase,
               req: raw_esi.ServiceImplementReqOp):
    self._req = req
    portReqsBlock = req.portReqs.blocks[0]

    # Find the output channel requests and store the settable proxies.
    num_output_ports = len(mod.outputs)
    to_client_reqs = [
        req for req in portReqsBlock
        if isinstance(req, raw_esi.ServiceImplementConnReqOp)
    ]
    self._output_reqs = [
        _OutputBundleSetter(req, self._req.results[num_output_ports + idx])
        for idx, req in enumerate(to_client_reqs)
    ]
    assert len(self._output_reqs) == len(req.results) - num_output_ports

  @property
  def reqs(self) -> List[NamedChannelValue]:
    """Get the list of incoming channels from the 'to server' connection
    requests."""
    return self._input_reqs

  @property
  def to_client_reqs(self) -> List[_OutputBundleSetter]:
    return self._output_reqs

  def check_unconnected_outputs(self):
    for req in self._output_reqs:
      if req._bundle_to_replace is not None:
        name_str = ".".join(req.client_name)
        raise ValueError(f"{name_str} has not been connected.")


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

  def generate_svc_impl(self,
                        serviceReq: raw_esi.ServiceImplementReqOp) -> bool:
    """"Generate the service inline and replace the `ServiceInstanceOp` which is
    being implemented."""

    assert len(self.generators) == 1
    generator: Generator = list(self.generators.values())[0]
    ports = self.generator_port_proxy(serviceReq.operation.operands, self)
    with self.GeneratorCtxt(self, ports, serviceReq, generator.loc):

      # Run the generator.
      bundles = _ServiceGeneratorBundles(self, serviceReq)
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
      serviceReq.operation.erase()

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
      ret = impl._builder.generate_svc_impl(serviceReq=req.opview)
    # The service implementation generator could have instantiated new modules,
    # so we need to generate them. Don't run the appID indexer since during a
    # pass, the IR can be invalid and the indexers assumes it is valid.
    sys.generate(skip_appid_index=True)
    return ret


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


@ServiceDecl
class MMIO:
  """ESI standard service to request access to an MMIO region."""

  read = Bundle([
      BundledChannel("offset", ChannelDirection.TO, Bits(32)),
      BundledChannel("data", ChannelDirection.FROM, Bits(32))
  ])

  @staticmethod
  def _op(sym_name: ir.StringAttr):
    return raw_esi.MMIOServiceDeclOp(sym_name)


class _FuncService(ServiceDecl):
  """ESI standard service to request execution of a function."""

  def __init__(self):
    super().__init__(self.__class__)

  def expose(self, name: AppID, arg_type: Type,
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


def package(sys: System):
  """Package all ESI collateral."""

  import shutil
  __root_dir__ = Path(__file__).parent

  # When pycde is installed through a proper install, all of the collateral
  # files are under a dir called "collateral".
  collateral_dir = __root_dir__ / "collateral"
  if collateral_dir.exists():
    esi_lib_dir = collateral_dir
  else:
    # Build we also want to allow pycde to work in-tree for developers. The
    # necessary files are screwn around the build tree.
    build_dir = __root_dir__.parents[4]
    circt_lib_dir = build_dir / "tools" / "circt" / "lib"
    esi_lib_dir = circt_lib_dir / "Dialect" / "ESI"
  # shutil.copy(esi_lib_dir / "ESIPrimitives.sv", sys.hw_output_dir)
