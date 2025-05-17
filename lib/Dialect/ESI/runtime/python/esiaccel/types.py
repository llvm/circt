# ===-----------------------------------------------------------------------===#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ===-----------------------------------------------------------------------===#
#
# The structure of the Python classes and hierarchy roughly mirrors the C++
# side, but wraps the C++ objects. The wrapper classes sometimes add convenience
# functionality and serve to return wrapped versions of the returned objects.
#
# ===-----------------------------------------------------------------------===#

from __future__ import annotations

from . import esiCppAccel as cpp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .accelerator import HWModule

from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import sys
import traceback


def _get_esi_type(cpp_type: cpp.Type):
  """Get the wrapper class for a C++ type."""
  for cpp_type_cls, fn in __esi_mapping.items():
    if isinstance(cpp_type, cpp_type_cls):
      return fn(cpp_type)
  return ESIType(cpp_type)


# Mapping from C++ types to functions constructing the Python object
# corresponding to that type.
__esi_mapping: Dict[Type, Callable] = {
    cpp.ChannelType: lambda cpp_type: _get_esi_type(cpp_type.inner)
}


class ESIType:

  def __init__(self, cpp_type: cpp.Type):
    self.cpp_type = cpp_type

  @property
  def supports_host(self) -> Tuple[bool, Optional[str]]:
    """Does this type support host communication via Python? Returns either
    '(True, None)' if it is, or '(False, reason)' if it is not."""

    if self.bit_width % 8 != 0:
      return (False, "runtime only supports types with multiple of 8 bits")
    return (True, None)

  def is_valid(self, obj) -> Tuple[bool, Optional[str]]:
    """Is a Python object compatible with HW type?  Returns either '(True,
    None)' if it is, or '(False, reason)' if it is not."""
    assert False, "unimplemented"

  @property
  def bit_width(self) -> int:
    """Size of this type, in bits. Negative for unbounded types."""
    assert False, "unimplemented"

  @property
  def max_size(self) -> int:
    """Maximum size of a value of this type, in bytes."""
    bitwidth = int((self.bit_width + 7) / 8)
    if bitwidth < 0:
      return bitwidth
    return bitwidth

  def serialize(self, obj) -> bytearray:
    """Convert a Python object to a bytearray."""
    assert False, "unimplemented"

  def deserialize(self, data: bytearray) -> Tuple[object, bytearray]:
    """Convert a bytearray to a Python object. Return the object and the
    leftover bytes."""
    assert False, "unimplemented"

  def __str__(self) -> str:
    return str(self.cpp_type)


class VoidType(ESIType):

  def is_valid(self, obj) -> Tuple[bool, Optional[str]]:
    if obj is not None:
      return (False, f"void type cannot must represented by None, not {obj}")
    return (True, None)

  @property
  def bit_width(self) -> int:
    return 8

  def serialize(self, obj) -> bytearray:
    # By convention, void is represented by a single byte of value 0.
    return bytearray([0])

  def deserialize(self, data: bytearray) -> Tuple[object, bytearray]:
    if len(data) == 0:
      raise ValueError(f"void type cannot be represented by {data}")
    return (None, data[1:])


__esi_mapping[cpp.VoidType] = VoidType


class BitsType(ESIType):

  def __init__(self, cpp_type: cpp.BitsType):
    self.cpp_type: cpp.BitsType = cpp_type

  def is_valid(self, obj) -> Tuple[bool, Optional[str]]:
    if not isinstance(obj, (bytearray, bytes, list)):
      return (False, f"invalid type: {type(obj)}")
    if isinstance(obj, list) and not all(
        [isinstance(b, int) and b.bit_length() <= 8 for b in obj]):
      return (False, f"list item too large: {obj}")
    if len(obj) != self.max_size:
      return (False, f"wrong size: {len(obj)}")
    return (True, None)

  @property
  def bit_width(self) -> int:
    return self.cpp_type.width

  def serialize(self, obj: Union[bytearray, bytes, List[int]]) -> bytearray:
    if isinstance(obj, bytearray):
      return obj
    if isinstance(obj, bytes) or isinstance(obj, list):
      return bytearray(obj)
    raise ValueError(f"cannot convert {obj} to bytearray")

  def deserialize(self, data: bytearray) -> Tuple[bytearray, bytearray]:
    return (data[0:self.max_size], data[self.max_size:])


__esi_mapping[cpp.BitsType] = BitsType


class IntType(ESIType):

  def __init__(self, cpp_type: cpp.IntegerType):
    self.cpp_type: cpp.IntegerType = cpp_type

  @property
  def bit_width(self) -> int:
    return self.cpp_type.width


class UIntType(IntType):

  def is_valid(self, obj) -> Tuple[bool, Optional[str]]:
    if not isinstance(obj, int):
      return (False, f"must be an int, not {type(obj)}")
    if obj < 0 or obj.bit_length() > self.bit_width:
      return (False, f"out of range: {obj}")
    return (True, None)

  def __str__(self) -> str:
    return f"uint{self.bit_width}"

  def serialize(self, obj: int) -> bytearray:
    return bytearray(int.to_bytes(obj, self.max_size, "little"))

  def deserialize(self, data: bytearray) -> Tuple[int, bytearray]:
    return (int.from_bytes(data[0:self.max_size],
                           "little"), data[self.max_size:])


__esi_mapping[cpp.UIntType] = UIntType


class SIntType(IntType):

  def is_valid(self, obj) -> Tuple[bool, Optional[str]]:
    if not isinstance(obj, int):
      return (False, f"must be an int, not {type(obj)}")
    if obj < 0:
      if (-1 * obj) > 2**(self.bit_width - 1):
        return (False, f"out of range: {obj}")
    elif obj < 0:
      if obj >= 2**(self.bit_width - 1) - 1:
        return (False, f"out of range: {obj}")
    return (True, None)

  def __str__(self) -> str:
    return f"sint{self.bit_width}"

  def serialize(self, obj: int) -> bytearray:
    return bytearray(int.to_bytes(obj, self.max_size, "little", signed=True))

  def deserialize(self, data: bytearray) -> Tuple[int, bytearray]:
    return (int.from_bytes(data[0:self.max_size], "little",
                           signed=True), data[self.max_size:])


__esi_mapping[cpp.SIntType] = SIntType


class StructType(ESIType):

  def __init__(self, cpp_type: cpp.StructType):
    self.cpp_type = cpp_type
    self.fields: List[Tuple[str, ESIType]] = [
        (name, _get_esi_type(ty)) for (name, ty) in cpp_type.fields
    ]

  @property
  def bit_width(self) -> int:
    widths = [ty.bit_width for (_, ty) in self.fields]
    if any([w < 0 for w in widths]):
      return -1
    return sum(widths)

  def is_valid(self, obj) -> Tuple[bool, Optional[str]]:
    fields_count = 0
    if not isinstance(obj, dict):
      obj = obj.__dict__

    for (fname, ftype) in self.fields:
      if fname not in obj:
        return (False, f"missing field '{fname}'")
      fvalid, reason = ftype.is_valid(obj[fname])
      if not fvalid:
        return (False, f"invalid field '{fname}': {reason}")
      fields_count += 1
    if fields_count != len(obj):
      return (False, "missing fields")
    return (True, None)

  def serialize(self, obj) -> bytearray:
    ret = bytearray()
    for (fname, ftype) in reversed(self.fields):
      fval = obj[fname]
      ret.extend(ftype.serialize(fval))
    return ret

  def deserialize(self, data: bytearray) -> Tuple[Dict[str, Any], bytearray]:
    ret = {}
    for (fname, ftype) in reversed(self.fields):
      (fval, data) = ftype.deserialize(data)
      ret[fname] = fval
    return (ret, data)


__esi_mapping[cpp.StructType] = StructType


class ArrayType(ESIType):

  def __init__(self, cpp_type: cpp.ArrayType):
    self.cpp_type = cpp_type
    self.element_type = _get_esi_type(cpp_type.element)
    self.size = cpp_type.size

  @property
  def bit_width(self) -> int:
    return self.element_type.bit_width * self.size

  def is_valid(self, obj) -> Tuple[bool, Optional[str]]:
    if not isinstance(obj, list):
      return (False, f"must be a list, not {type(obj)}")
    if len(obj) != self.size:
      return (False, f"wrong size: expected {self.size} not {len(obj)}")
    for (idx, e) in enumerate(obj):
      evalid, reason = self.element_type.is_valid(e)
      if not evalid:
        return (False, f"invalid element {idx}: {reason}")
    return (True, None)

  def serialize(self, lst: list) -> bytearray:
    ret = bytearray()
    for e in reversed(lst):
      ret.extend(self.element_type.serialize(e))
    return ret

  def deserialize(self, data: bytearray) -> Tuple[List[Any], bytearray]:
    ret = []
    for _ in range(self.size):
      (obj, data) = self.element_type.deserialize(data)
      ret.append(obj)
    ret.reverse()
    return (ret, data)


__esi_mapping[cpp.ArrayType] = ArrayType


class Port:
  """A unidirectional communication channel. This is the basic communication
  method with an accelerator."""

  def __init__(self, owner: BundlePort, cpp_port: cpp.ChannelPort):
    self.owner = owner
    self.cpp_port = cpp_port
    self.type = _get_esi_type(cpp_port.type)

  def connect(self, buffer_size: Optional[int] = None):
    (supports_host, reason) = self.type.supports_host
    if not supports_host:
      raise TypeError(f"unsupported type: {reason}")

    self.cpp_port.connect(buffer_size)
    return self

  def disconnect(self):
    self.cpp_port.disconnect()


class WritePort(Port):
  """A unidirectional communication channel from the host to the accelerator."""

  def __init__(self, owner: BundlePort, cpp_port: cpp.WriteChannelPort):
    super().__init__(owner, cpp_port)
    self.cpp_port: cpp.WriteChannelPort = cpp_port

  def __serialize_msg(self, msg=None) -> bytearray:
    valid, reason = self.type.is_valid(msg)
    if not valid:
      raise ValueError(
          f"'{msg}' cannot be converted to '{self.type}': {reason}")
    msg_bytes: bytearray = self.type.serialize(msg)
    return msg_bytes

  def write(self, msg=None) -> bool:
    """Write a typed message to the channel. Attempts to serialize 'msg' to what
        the accelerator expects, but will fail if the object is not convertible to
        the port type."""
    self.cpp_port.write(self.__serialize_msg(msg))
    return True

  def try_write(self, msg=None) -> bool:
    """Like 'write', but uses the non-blocking tryWrite method of the underlying
        port. Returns True if the write was successful, False otherwise."""
    return self.cpp_port.tryWrite(self.__serialize_msg(msg))


class ReadPort(Port):
  """A unidirectional communication channel from the accelerator to the host."""

  def __init__(self, owner: BundlePort, cpp_port: cpp.ReadChannelPort):
    super().__init__(owner, cpp_port)
    self.cpp_port: cpp.ReadChannelPort = cpp_port

  def read(self) -> object:
    """Read a typed message from the channel. Returns a deserialized object of a
    type defined by the port type."""

    buffer = self.cpp_port.read()
    (msg, leftover) = self.type.deserialize(buffer)
    if len(leftover) != 0:
      raise ValueError(f"leftover bytes: {leftover}")
    return msg


class BundlePort:
  """A collections of named, unidirectional communication channels."""

  # When creating a new port, we need to determine if it is a service port and
  # instantiate it correctly.
  def __new__(cls, owner: HWModule, cpp_port: cpp.BundlePort):
    # TODO: add a proper registration mechanism for service ports.
    if isinstance(cpp_port, cpp.Function):
      return super().__new__(FunctionPort)
    if isinstance(cpp_port, cpp.Callback):
      return super().__new__(CallbackPort)
    if isinstance(cpp_port, cpp.MMIORegion):
      return super().__new__(MMIORegion)
    if isinstance(cpp_port, cpp.Telemetry):
      return super().__new__(TelemetryPort)
    return super().__new__(cls)

  def __init__(self, owner: HWModule, cpp_port: cpp.BundlePort):
    self.owner = owner
    self.cpp_port = cpp_port

  def write_port(self, channel_name: str) -> WritePort:
    return WritePort(self, self.cpp_port.getWrite(channel_name))

  def read_port(self, channel_name: str) -> ReadPort:
    return ReadPort(self, self.cpp_port.getRead(channel_name))


class MessageFuture(Future):
  """A specialization of `Future` for ESI messages. Wraps the cpp object and
  deserializes the result.  Hopefully overrides all the methods necessary for
  proper operation, which is assumed to be not all of them."""

  def __init__(self, result_type: Type, cpp_future: cpp.MessageDataFuture):
    self.result_type = result_type
    self.cpp_future = cpp_future

  def running(self) -> bool:
    return True

  def done(self) -> bool:
    return self.cpp_future.valid()

  def result(self, timeout: Optional[Union[int, float]] = None) -> Any:
    # TODO: respect timeout
    self.cpp_future.wait()
    result_bytes = self.cpp_future.get()
    (msg, leftover) = self.result_type.deserialize(result_bytes)
    if len(leftover) != 0:
      raise ValueError(f"leftover bytes: {leftover}")
    return msg

  def add_done_callback(self, fn: Callable[[Future], object]) -> None:
    raise NotImplementedError("add_done_callback is not implemented")


class MMIORegion(BundlePort):
  """A region of memory-mapped I/O space. This is a collection of named
  channels, which are either read or read-write. The channels are accessed
  by name, and can be connected to the host."""

  def __init__(self, owner: HWModule, cpp_port: cpp.MMIORegion):
    super().__init__(owner, cpp_port)
    self.region = cpp_port

  @property
  def descriptor(self) -> cpp.MMIORegionDesc:
    return self.region.descriptor

  def read(self, offset: int) -> bytearray:
    """Read a value from the MMIO region at the given offset."""
    return self.region.read(offset)

  def write(self, offset: int, data: bytearray) -> None:
    """Write a value to the MMIO region at the given offset."""
    self.region.write(offset, data)


class FunctionPort(BundlePort):
  """A pair of channels which carry the input and output of a function."""

  def __init__(self, owner: HWModule, cpp_port: cpp.BundlePort):
    super().__init__(owner, cpp_port)
    self.arg_type = self.write_port("arg").type
    self.result_type = self.read_port("result").type
    self.connected = False

  def connect(self):
    self.cpp_port.connect()
    self.connected = True

  def call(self, **kwargs: Any) -> Future:
    """Call the function with the given argument and returns a future of the
    result."""
    valid, reason = self.arg_type.is_valid(kwargs)
    if not valid:
      raise ValueError(
          f"'{kwargs}' cannot be converted to '{self.arg_type}': {reason}")
    arg_bytes: bytearray = self.arg_type.serialize(kwargs)
    cpp_future = self.cpp_port.call(arg_bytes)
    return MessageFuture(self.result_type, cpp_future)

  def __call__(self, *args: Any, **kwds: Any) -> Future:
    return self.call(*args, **kwds)


class CallbackPort(BundlePort):
  """Callback ports are the inverse of function ports -- instead of calls to the
  accelerator, they get called from the accelerator. Specify the function which
  you'd like the accelerator to call when you call `connect`."""

  def __init__(self, owner: HWModule, cpp_port: cpp.BundlePort):
    super().__init__(owner, cpp_port)
    self.arg_type = self.read_port("arg").type
    self.result_type = self.write_port("result").type
    self.connected = False

  def connect(self, cb: Callable[[Any], Any]):

    def type_convert_wrapper(cb: Callable[[Any], Any],
                             msg: bytearray) -> Optional[bytearray]:
      try:
        (obj, leftover) = self.arg_type.deserialize(msg)
        if len(leftover) != 0:
          raise ValueError(f"leftover bytes: {leftover}")
        result = cb(obj)
        if result is None:
          return None
        return self.result_type.serialize(result)
      except Exception as e:
        traceback.print_exception(e)
        return None

    self.cpp_port.connect(lambda x: type_convert_wrapper(cb=cb, msg=x))
    self.connected = True


class TelemetryPort(BundlePort):
  """Telemetry ports report an individual piece of information from the
  acceelerator. The method of accessing telemetry will likely change in the
  future."""

  def __init__(self, owner: HWModule, cpp_port: cpp.BundlePort):
    super().__init__(owner, cpp_port)
    self.connected = False

  def connect(self):
    self.cpp_port.connect()
    self.connected = True

  def read(self) -> Future:
    cpp_future = self.cpp_port.read()
    return MessageFuture(self.cpp_port.type, cpp_future)
