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
                    self._root._get_chan_offset_bitcount(True, client_path),
                    self._root._get_chan_offset_bitcount(False, client_path),
                    read_type, write_type)


class Xrt(_XrtNode):

  def __init__(self,
               xclbin: os.PathLike = None,
               kernel: str = None,
               chan_desc_path: os.PathLike = None,
               hw_emu: bool = False) -> None:
    if xclbin is None:
      xclbin_files = list(__dir__.glob("*.xclbin"))
      if len(xclbin_files) == 0:
        raise RuntimeError("Could not find FPGA image.")
      if len(xclbin_files) > 1:
        raise RuntimeError("Found multiple FPGA images.")
      xclbin = __dir__ / xclbin_files[0]
    if kernel is None:
      xclbin_fn = os.path.basename(xclbin)
      kernel = xclbin_fn.split('.')[0]
    if chan_desc_path is None:
      chan_desc_path = __dir__ / "xrt_mmio_descriptor.json"
    self.chan_desc = json.loads(open(chan_desc_path).read())
    super().__init__(self, [])

    if hw_emu:
      os.environ["XCL_EMULATION_MODE"] = "hw_emu"

    from .esiXrtPython import Accelerator
    self._acc = Accelerator(os.path.abspath(str(xclbin)), kernel)

  def _get_chan_offset_bitcount(
      self, from_host: bool,
      client_path: typing.List[str]) -> typing.Tuple[int, int]:
    if from_host:
      for channel in self.chan_desc["from_host_regs"]:
        if channel["client_path"] == client_path:
          return (channel["offset"], channel["size"])
    else:
      for channel in self.chan_desc["to_host_regs"]:
        if channel["client_path"] == client_path:
          return (channel["offset"], channel["size"])
    return (None, None)


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

  def __init__(self, acc, from_host_chan_desc: typing.Tuple[int, int],
               to_host_chan_desc: typing.Tuple[int, int],
               read_type: typing.Optional[Type],
               write_type: typing.Optional[Type]):
    self._acc = acc
    self.from_host_chan_offset, self.from_host_chan_size = from_host_chan_desc
    self.to_host_chan_offset, self.to_host_chan_size = to_host_chan_desc
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
    if self.from_host_chan_offset is None:
      raise RuntimeError("This port doesn't have a write channel")
    enc_msg = self._write_convert.write(msg)
    self._acc.send_msg(self.from_host_chan_offset, self.from_host_chan_size,
                       enc_msg)
    return True

  def read(self, blocking_time: typing.Optional[float]):
    """Read a message from this port. If 'blocking_timeout' is None, return
    immediately. Otherwise, wait up to 'blocking_timeout' for a message. Returns
    the message if found, None if no message was read."""

    if self.to_host_chan_offset is None:
      raise RuntimeError("This port doesn't have a read channel")

    if blocking_time is None:
      # Non-blocking.
      recvResp = self._acc.recv_msg(self.to_host_chan_offset,
                                    self.to_host_chan_size)
    else:
      # Blocking. Since our cosim rpc server doesn't currently support blocking
      # reads, use polling instead.
      e = time.time() + blocking_time
      recvResp = None
      while recvResp is None or e > time.time():
        recvResp = self._acc.recv_msg(self.to_host_chan_offset,
                                      self.to_host_chan_size)
        if recvResp is not None:
          break
        else:
          time.sleep(0.001)
    if recvResp is None:
      return None
    return self._read_convert.read(recvResp)
