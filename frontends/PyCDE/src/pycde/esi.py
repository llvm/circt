from pycde.value import ChannelValue
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
