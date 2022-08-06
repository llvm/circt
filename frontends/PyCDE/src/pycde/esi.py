from pycde.system import System
from .module import Generator, _module_base, _SpecializedModule
from pycde.value import ChannelValue, Value
from .common import AppID, Input, Output, InputChannel, OutputChannel, _PyProxy
from circt.dialects import esi as raw_esi, hw
from pycde.pycde_types import ChannelType, PyCDEType, types

import mlir.ir as ir

from typing import Callable, Dict, List, Tuple, Type, Union

from pycde import support

ToServer = InputChannel
FromServer = OutputChannel


class ServiceDecl(_PyProxy):
  """Declare an ESI service interface."""

  def __init__(self, cls: Type):
    self.name = cls.__name__
    for (attr_name, attr) in cls.__dict__.items():
      if isinstance(attr, InputChannel):
        setattr(self, attr_name, _RequestToServerConn(self, attr.type,
                                                      attr_name))
      elif isinstance(attr, OutputChannel):
        setattr(self, attr_name, _RequestToClientConn(self, attr.type,
                                                      attr_name))
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
            raw_esi.ToServerOp(attr._name, ir.TypeAttr.get(attr.type))
          elif isinstance(attr, _RequestToClientConn):
            raw_esi.ToClientOp(attr._name, ir.TypeAttr.get(attr.type))
    return sym_name


class _RequestConnection:
  """Parent to 'request' proxy classes. Constructed as attributes on the
  ServiceDecl class. Provides syntactic sugar for constructing service
  connection requests."""

  def __init__(self, decl: ServiceDecl, type: PyCDEType, attr_name: str):
    self.decl = decl
    self._name = ir.StringAttr.get(attr_name)
    self.type = type


class _RequestToServerConn(_RequestConnection):

  def __call__(self, chan: ChannelValue, chan_name: str):
    decl_sym = self.decl._materialize_service_decl()
    raw_esi.RequestToServerConnectionOp(
        hw.InnerRefAttr.get(ir.StringAttr.get(decl_sym), self._name),
        chan.value, ir.ArrayAttr.get([ir.StringAttr.get(chan_name)]))


class _RequestToClientConn(_RequestConnection):

  def __call__(self, type: PyCDEType, chan_name: str):
    decl_sym = self.decl._materialize_service_decl()
    if not isinstance(type, ChannelType):
      type = types.channel(type)
    req_op = raw_esi.RequestToClientConnectionOp(
        type, hw.InnerRefAttr.get(ir.StringAttr.get(decl_sym), self._name),
        ir.ArrayAttr.get([ir.StringAttr.get(chan_name)]))
    return ChannelValue(req_op)


@ServiceDecl
class HostComms:
  to_host = ToServer(types.any)
  from_host = FromServer(types.any)


def Cosim(decl: ServiceDecl, clk, rst):

  # TODO: better modeling and implementation capacity. The below is just
  # temporary.
  raw_esi.ServiceInstanceOp(result=[],
                            service_symbol=ir.FlatSymbolRefAttr.get(
                                decl._materialize_service_decl()),
                            impl_type=ir.StringAttr.get("cosim"),
                            inputs=[clk.value, rst.value])


def ServiceImplementation(decl: ServiceDecl):

  def wrap(service_impl, decl: ServiceDecl = decl):

    def instantiate_cb(mod: _SpecializedModule, instance_name: str,
                       inputs: dict, appid: AppID, loc):
      opts = _service_generator_registry.register(mod)
      return raw_esi.ServiceInstanceOp(
          result=[t for _, t in mod.output_ports],
          service_symbol=ir.FlatSymbolRefAttr.get(
              decl._materialize_service_decl()),
          impl_type=_ServiceGeneratorRegistry._impl_type_name,
          inputs=[inputs[pn].value for pn, _ in mod.input_ports],
          impl_opts=opts)

    def generate_cb(generator: Generator, spec_mod: _SpecializedModule,
                    extra_args: Dict[str, object]):
      # return _generate_block(generator, spec_mod, None, extra_args)
      pass

    return _module_base(service_impl,
                        extern_name=None,
                        generator_cb=generate_cb,
                        instantiate_cb=instantiate_cb)

  return wrap


class _ServiceGeneratorRegistry:
  """Class to register individual service instance generators."""
  _registered = False
  _impl_type_name = ir.StringAttr.get("pycde")

  def __init__(self):
    self._registry = {}
    assert _ServiceGeneratorRegistry._registered is False, \
      "Cannot instantiate more than one _ServiceGeneratorRegistry"
    raw_esi.registerServiceGenerator(
        _ServiceGeneratorRegistry._impl_type_name.value,
        self._implement_service)
    _ServiceGeneratorRegistry._registered = True

  def register(self, service_implementation: _SpecializedModule) -> ir.DictAttr:
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
    """This is the callback which the ESI connect-services pass calls."""
    assert isinstance(req.opview, raw_esi.ServiceImplementReqOp)
    opts = ir.DictAttr(req.attributes["impl_opts"])
    impl_name = opts["name"]
    if impl_name not in self._registry:
      return False
    (impl, sys) = self._registry[impl_name]
    with sys:
      return impl.generate(input_channels=None, output_channels=None)


_service_generator_registry = _ServiceGeneratorRegistry()
