#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from multiprocessing.sharedctypes import Value
import capnp
import time
import typing


class Type:

  def __init__(self, type_id: typing.Optional[int] = None):
    self.type_id = type_id

  def is_valid(self, obj):
    """Is a Python object compatible with HW type."""
    assert False, "unimplemented"


class IntType(Type):

  def __init__(self,
               width: int,
               signed: bool,
               type_id: typing.Optional[int] = None):
    super().__init__(type_id)
    self.width = width
    self.signed = signed

  def is_valid(self, obj):
    if not isinstance(obj, int):
      return False
    if obj >= 2**self.width:
      return False
    return True

  def __str__(self):
    return ("" if self.signed else "u") + \
      f"int{self.width}"


class Port:

  def __init__(self,
               client_path: typing.List[str],
               backend,
               read_type: typing.Optional[Type] = None,
               write_type: typing.Optional[Type] = None):
    self._backend = backend.get_port(client_path, read_type, write_type)
    self.client_path = client_path
    self.read_type = read_type
    self.write_type = write_type


class WritePort(Port):

  def write(self, msg) -> bool:
    assert self.write_type is not None, "Expected non-None write_type"
    if not self.write_type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.write_type}'")
    return self._backend.write(msg)


class ReadPort(Port):

  def read(self, blocking_timeout: typing.Optional[float] = 1.0):
    return self._backend.read(blocking_timeout)


class ReadWritePort(Port):

  def write(self, msg) -> bool:
    assert self.write_type is not None, "Expected non-None write_type"
    if not self.write_type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.write_type}'")
    return self._backend.write(msg)

  def read(self, blocking_timeout: typing.Optional[float] = 1.0):
    return self._backend.read(blocking_timeout)


class _CosimNode:
  """Provides a capnp-based co-simulation backend."""

  def __init__(self, root, prefix: typing.List[str]):
    self._root: Cosim = root
    self._endpoint_prefix = prefix

  def get_child(self, child_name: str):
    """When instantiating a child instance, get the backend node with which it
    is associated."""
    child_path = self._endpoint_prefix + [child_name]
    return _CosimNode(self._root, child_path)

  def get_port(self,
               client_path: typing.List[str],
               read_type: typing.Optional[Type] = None,
               write_type: typing.Optional[Type] = None):
    """When building a service port, get the backend port which it should use
    for interactions."""
    path = ".".join(self._endpoint_prefix) + "." + "_".join(client_path)
    ep = self._root._open_endpoint(
        path,
        write_type=write_type.type_id if write_type is not None else None,
        read_type=read_type.type_id if read_type is not None else None)
    return _CosimPort(self, ep, read_type, write_type)


class Cosim(_CosimNode):
  """Connect to a Cap'N Proto RPC co-simulation and provide a cosim backend
  service."""

  def __init__(self, schemaPath, hostPort):
    """Load the schema and connect to the RPC server"""
    self._schema = capnp.load(schemaPath)
    self._rpc_client = capnp.TwoPartyClient(hostPort)
    self._cosim = self._rpc_client.bootstrap().cast_as(
        self._schema.CosimDpiServer)

    # Find the simulation prefix and use it in our parent constructor.
    ifaces = self.list()
    prefix = [] if len(ifaces) == 0 else ifaces[0].endpointID.split(".")[:1]
    super().__init__(self, prefix)

  def list(self):
    """List the available interfaces"""
    return self._cosim.list().wait().ifaces

  def _open_endpoint(self, epid: str, write_type=None, read_type=None):
    """Open the endpoint, optionally checking the send and recieve types"""
    for iface in self.list():
      if iface.endpointID == epid:
        # Optionally check that the type IDs match.
        if write_type is not None:
          assert iface.sendTypeID == write_type.schema.node.id
        else:
          assert write_type is None
        if read_type is not None:
          assert iface.recvTypeID == read_type.schema.node.id
        else:
          assert read_type is None

        openResp = self._cosim.open(iface).wait()
        assert openResp.iface is not None
        return openResp.iface
    assert False, f"Could not find specified EndpointID: {epid}"


class _CosimPort:
  """Cosim backend for service ports. This is where the real meat is buried."""

  class _TypeConverter:
    """Parent class for Capnp type converters."""

    def __init__(self, schema, esi_type: Type):
      self.esi_type = esi_type
      assert hasattr(esi_type, "capnp_name")
      if not hasattr(schema, esi_type.capnp_name):
        raise ValueError("Cosim does not support non-capnp types.")
      self.capnp_type = getattr(schema, esi_type.capnp_name)

  class _IntConverter(_TypeConverter):
    """Convert python ints to and from capnp messages."""

    def write(self, py_int: int):
      return self.capnp_type.new_message(i=py_int)

    def read(self, capnp_resp) -> int:
      return capnp_resp.as_struct(self.capnp_type).i

  # Lookup table for getting the correct type converter for a given type.
  ConvertLookup = {IntType: _IntConverter}

  def __init__(self, node: _CosimNode, endpoint,
               read_type: typing.Optional[Type],
               write_type: typing.Optional[Type]):
    self._endpoint = endpoint
    schema = node._root._schema
    # For each type, lookup the type converter and store that instead of the
    # type itself.
    if read_type is not None:
      converter = _CosimPort.ConvertLookup[type(read_type)]
      self._read_convert = converter(schema, read_type)
    if write_type is not None:
      converter = _CosimPort.ConvertLookup[type(write_type)]
      self._write_convert = converter(schema, write_type)

  def write(self, msg) -> bool:
    """Write a message to this port."""
    self._endpoint.send(self._write_convert.write(msg))
    return True

  def read(self, blocking_time: typing.Optional[float]):
    """Read a message from this port. If 'blocking_timeout' is None, return
    immediately. Otherwise, wait up to 'blocking_timeout' for a message. Returns
    the message if found, None if no message was read."""

    if blocking_time is None:
      # Non-blocking.
      recvResp = self._endpoint.recv(False).wait()
    else:
      # Blocking. Since our cosim rpc server doesn't currently support blocking
      # reads, use polling instead.
      e = time.time() + blocking_time
      recvResp = None
      while recvResp is None or e > time.time():
        recvResp = self._endpoint.recv(False).wait()
        if recvResp.hasData:
          break
        else:
          time.sleep(0.001)
    if not recvResp.hasData:
      return None
    assert recvResp.resp is not None
    return self._read_convert.read(recvResp.resp)
