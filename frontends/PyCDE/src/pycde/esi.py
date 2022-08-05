from pycde.value import ChannelValue, Value
from .common import Input, Output, InputChannel, OutputChannel, _PyProxy
from circt.dialects import esi as raw_esi, hw
from pycde.pycde_types import ChannelType, PyCDEType, types

import mlir.ir as ir

import typing

ToServer = InputChannel
FromServer = OutputChannel


class ServiceDecl(_PyProxy):
  """Declare an ESI service interface."""

  def __init__(self, cls: typing.Type):
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


class ServiceInstanceOp:

  def __init__(self, result: typing.List[PyCDEType],
               service_symbol: ir.FlatSymbolRefAttr, impl_type: ir.Attribute,
               inputs: typing.List[Value],
               output_port_lookup: typing.Dict[str, int]):
    self._op = raw_esi.ServiceInstanceOp(result, service_symbol, impl_type,
                                         inputs)
    self._output_port_lookup = output_port_lookup

  def __getattr__(self, name: str) -> object:
    if (name in self._output_port_lookup):
      return Value(self._op.results[self._output_port_lookup[name][0]])
    raise AttributeError(name)


def ServiceImplementation(decl: ServiceDecl):

  class ServiceImplementation:

    def __init__(self, service_impl_class):
      self._service_impl_class = service_impl_class
      self._impl_name = ir.StringAttr.get("pycde:" +
                                          service_impl_class.__name__)
      output_ports = [
          pn for pn in [(name, getattr(service_impl_class, name))
                        for name in dir(service_impl_class)]
          if isinstance(pn[1], Output)
      ]
      self._output_port_lookup = {
          pn[0]: (port_num, pn[1]) for (port_num, pn) in enumerate(output_ports)
      }

      input_ports = [
          pn for pn in [(name, getattr(service_impl_class, name))
                        for name in dir(service_impl_class)]
          if isinstance(pn[1], Input)
      ]
      self._input_port_lookup = {
          pn[0]: (port_num, pn[1]) for (port_num, pn) in enumerate(input_ports)
      }

    def __call__(self, **kwargs):
      inputs_missing = set(self._input_port_lookup.keys()) - set(kwargs.keys())
      if inputs_missing:
        raise ValueError(f"Missing inputs: {inputs_missing}")

      inputs = [None] * len(self._input_port_lookup)
      for (k, v) in kwargs.items():
        if k not in self._input_port_lookup:
          raise ValueError(f"Unknown input port: {k}")
        port_num, port_type = self._input_port_lookup[k]
        if port_type.type != v.type:
          raise ValueError(f"Type mismatch for input port {k}: "
                           f"expected {port_type.type}, got {v.type}")
        inputs[port_num] = v.value

      return ServiceInstanceOp(
          result=[t.type for _, t in self._output_port_lookup.values()],
          service_symbol=ir.FlatSymbolRefAttr.get(
              decl._materialize_service_decl()),
          impl_type=self._impl_name,
          inputs=inputs,
          output_port_lookup=self._output_port_lookup)

  return ServiceImplementation
