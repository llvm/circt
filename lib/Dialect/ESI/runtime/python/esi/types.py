#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import esiCppAccel as cpp

from typing import Callable, Dict, Optional, Tuple, Type


def _get_esi_type(cpp_type: cpp.Type):
  # if cpp_type in __esi_mapping:
  #   return __esi_mapping[type(cpp_type)](cpp_type)
  for cpp_type_cls, fn in __esi_mapping.items():
    if isinstance(cpp_type, cpp_type_cls):
      return fn(cpp_type)
  return ESIType(cpp_type)


__esi_mapping: Dict[Type, Callable] = {
    cpp.ChannelType: lambda cpp_type: _get_esi_type(cpp_type.inner)
}


class ESIType:

  def __init__(self, cpp_type: cpp.Type):
    self.cpp_type = cpp_type

  def is_valid(self, obj) -> bool:
    """Is a Python object compatible with HW type?"""
    assert False, "unimplemented"

  @property
  def max_size(self) -> int:
    """Maximum size of a value of this type, in bytes."""
    assert False, "unimplemented"

  def serialize(self, obj) -> bytearray:
    """Convert a Python object to a bytearray."""
    assert False, "unimplemented"

  def deserialize(self, data: bytearray) -> object:
    """Convert a bytearray to a Python object."""
    assert False, "unimplemented"

  def __str__(self) -> str:
    return str(self.cpp_type)


class VoidType(ESIType):

  def is_valid(self, obj) -> bool:
    return obj is None

  @property
  def max_size(self) -> int:
    return 1

  def serialize(self, obj) -> bytearray:
    # By convention, void is represented by a single byte of value 0.
    return bytearray([0])

  def deserialize(self, data: bytearray) -> object:
    if len(data) != 1:
      raise ValueError(f"void type cannot be represented by {data}")
    return None


__esi_mapping[cpp.VoidType] = VoidType


class BitsType(ESIType):

  def __init__(self, cpp_type: cpp.BitsType):
    self.cpp_type = cpp_type

  def is_valid(self, obj) -> bool:
    return isinstance(obj, bytearray) and len(obj) == int(
        (self.cpp_type.width + 7) / 8)

  @property
  def max_size(self) -> int:
    return int((self.cpp_type.width + 7) / 8)

  def serialize(self, obj) -> bytearray:
    return obj

  def deserialize(self, data: bytearray) -> object:
    return data


__esi_mapping[cpp.BitVectorType] = BitsType


class IntType(ESIType):

  def __init__(self, cpp_type: cpp.IntegerType):
    self.cpp_type = cpp_type

  @property
  def width(self) -> int:
    return self.cpp_type.width


__esi_mapping[cpp.IntegerType] = IntType


class UIntType(IntType):

  def is_valid(self, obj) -> bool:
    if not isinstance(obj, int):
      return False
    if obj < 0 or obj >= 2**self.width:
      return False
    return True

  def __str__(self) -> str:
    return f"uint{self.width}"


__esi_mapping[cpp.UIntType] = IntType


class SIntType(IntType):

  def is_valid(self, obj) -> bool:
    if not isinstance(obj, int):
      return False
    if obj < 0:
      if obj < 2**(self.width - 1):
        return False
    elif obj < 0:
      if obj >= 2**self.width:
        return False
    return True

  def __str__(self) -> str:
    return f"sint{self.width}"


__esi_mapping[cpp.SIntType] = IntType


class StructType(ESIType):

  def __init__(self, cpp_type: cpp.StructType):
    self.cpp_type = cpp_type

  def is_valid(self, obj) -> bool:
    fields_count = 0
    if not isinstance(obj, dict):
      obj = obj.__dict__

    for (fname, ftype) in self.cpp_type.fields:
      if fname not in obj:
        return False
      if not ftype.is_valid(obj[fname]):
        return False
      fields_count += 1
    if fields_count != len(obj):
      return False
    return True


class Port:

  def __init__(self, cpp_port: cpp.ChannelPort):
    self.cpp_port = cpp_port
    self.type = _get_esi_type(cpp_port.type)

  def connect(self):
    self.cpp_port.connect()
    return self


class WritePort(Port):

  def __init__(self, cpp_port: cpp.WriteChannelPort):
    super().__init__(cpp_port)
    self.cpp_port = cpp_port

  def write(self, msg=None) -> bool:
    if not self.type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.type}'")
    msg_bytes: bytearray = self.type.serialize(msg)
    self.cpp_port.write(msg_bytes)
    return True


class ReadPort(Port):

  def __init__(self, cpp_port: cpp.ReadChannelPort):
    super().__init__(cpp_port)
    self.cpp_port = cpp_port

  def read(self) -> Tuple[bool, Optional[object]]:
    """Read a message from the channel."""
    msg_bytes = self.cpp_port.read(self.type.max_size)
    if len(msg_bytes) == 0:
      return (False, None)
    return (True, self.type.deserialize(msg_bytes))


class BundlePort:

  def __init__(self, cpp_port: cpp.BundlePort):
    self.cpp_port = cpp_port

  def write_port(self, channel_name: str) -> WritePort:
    return WritePort(self.cpp_port.getWrite(channel_name))

  def read_port(self, channel_name: str) -> ReadPort:
    return ReadPort(self.cpp_port.getRead(channel_name))
