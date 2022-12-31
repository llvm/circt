from .common import *

import json
import os
from pathlib import Path
import time
import typing

__dir__ = Path(__file__).parent


class _XrtNode:

  def __init__(self, root, prefix: typing.List[str]):
    self._root: Xrt = root
    self._endpoint_prefix = prefix

  def supports_impl(self, impl_type: str) -> bool:
    """The cosim backend only supports cosim connectivity implementations."""
    return impl_type == "pycde"

  def get_child(self, child_name: str):
    """When instantiating a child instance, get the backend node with which it
    is associated."""
    child_path = self._endpoint_prefix + [child_name]
    return _XrtNode(self._root, child_path)

  def get_port(self,
               client_path: typing.List[str],
               read_type: typing.Optional[Type] = None,
               write_type: typing.Optional[Type] = None):
    """When building a service port, get the backend port which it should use
    for interactions."""

    return _XrtPort(self._root._acc,
                    self._root._get_chan_offset_bitcount(client_path),
                    read_type, write_type)


class Xrt(_XrtNode):

  def __init__(self,
               xclbin: os.PathLike,
               chan_desc_path: os.PathLike = None,
               hw_emu: bool = False) -> None:
    if chan_desc_path is None:
      chan_desc_path = __dir__ / "xrt_mmio_descriptor.json"
    self.chan_desc = json.loads(open(chan_desc_path).read())
    super().__init__(self, [])

    if hw_emu:
      os.environ["XCL_EMULATION_MODE"] = "hw_emu"

    from .esiXrtPython import Accelerator
    self._acc = Accelerator(os.path.abspath(str(xclbin)))

  def _get_chan_offset_bitcount(
      self, client_path: typing.List[str]) -> typing.Tuple[int, int]:
    for channel in self.chan_desc["from_host_regs"] + self.chan_desc[
        "to_host_regs"]:
      if channel["client_path"] == client_path:
        return (channel["offset"], channel["size"])
    raise ValueError(f"Could not find channel description for {client_path}")


class _XrtPort:
  """XRT backend for service ports. This is where the real meat is buried."""

  class _TypeConverter:
    """Parent class for Capnp type converters."""

    def __init__(self, esi_type: Type):
      self.esi_type = esi_type

  class _VoidConverter(_TypeConverter):
    """Convert python ints to and from capnp messages."""

    def write(self, py_int: None) -> int:
      return 0

    def read(self, resp: int) -> None:
      return

  class _IntConverter(_TypeConverter):
    """Convert python ints to and from binary."""

    def __init__(self, esi_type: IntType):
      super().__init__(esi_type)
      mask = 0
      # TODO: signed ints
      for _ in range(esi_type.width):
        mask = ((mask << 1) | 1)
      self.mask = mask

    def write(self, py_int: int) -> int:
      return py_int & self.mask

    def read(self, resp: int) -> int:
      return resp & self.mask

  class _StructConverter(_TypeConverter):
    """Convert python ints to and from capnp messages."""

    def __init__(self, esi_type: StructType):
      super().__init__(esi_type)
      self.fields: typing.List[str, typing.Tuple[int, callable]] = [
          (fname, (ftype.width, _XrtPort.ConvertLookup[type(ftype)](esi_type)))
          for (fname, ftype) in esi_type.fields
      ]

    def write(self, py_dict: dict):
      ret = 0
      for (fname, (width, fconv)) in self.fields:
        ret = (ret << width) | fconv.write(py_dict[fname])
      return ret

    def read(self, resp: int) -> typing.Dict:
      ret = {}
      for (fname, (width, fconv)) in reversed(self.fields):
        ret[fname] = fconv.read(resp)
        resp = resp >> width
      return ret

  # Lookup table for getting the correct type converter for a given type.
  ConvertLookup = {
      VoidType: _VoidConverter,
      IntType: _IntConverter,
      StructType: _StructConverter
  }

  def __init__(self, acc, chan_desc: typing.Tuple[int, int],
               read_type: typing.Optional[Type],
               write_type: typing.Optional[Type]):
    self._acc = acc
    self.chan_offset, self.chan_size = chan_desc
    # For each type, lookup the type converter and store that instead of the
    # type itself.
    if read_type is not None:
      converter = _XrtPort.ConvertLookup[type(read_type)]
      self._read_convert = converter(read_type)
    if write_type is not None:
      converter = _XrtPort.ConvertLookup[type(write_type)]
      self._write_convert = converter(write_type)

  def write(self, msg) -> bool:
    """Write a message to this port."""
    enc_msg = self._write_convert.write(msg)
    self._acc.send_msg(self.chan_offset, self.chan_size, enc_msg)
    return True

  def read(self, blocking_time: typing.Optional[float]):
    """Read a message from this port. If 'blocking_timeout' is None, return
    immediately. Otherwise, wait up to 'blocking_timeout' for a message. Returns
    the message if found, None if no message was read."""

    if blocking_time is None:
      # Non-blocking.
      recvResp = self._acc.recv_msg(self.chan_offset, self.chan_size)
    else:
      # Blocking. Since our cosim rpc server doesn't currently support blocking
      # reads, use polling instead.
      e = time.time() + blocking_time
      recvResp = None
      while recvResp is None or e > time.time():
        recvResp = self._acc.recv_msg(self.chan_offset, self.chan_size)
        if recvResp is not None:
          break
        else:
          time.sleep(0.001)
    if recvResp is None:
      return None
    return self._read_convert.read(recvResp)
