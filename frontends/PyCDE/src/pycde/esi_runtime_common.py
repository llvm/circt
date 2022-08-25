#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
               read_type: typing.Optional[Type] = None,
               write_type: typing.Optional[Type] = None):
    self.client_path = client_path
    self.read_type = read_type
    self.write_type = write_type


class WritePort(Port):

  def write(self, msg):
    assert self.write_type is not None, "Expected non-None write_type"
    if not self.write_type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.write_type}'")
    assert False, "Unimplemented"


class ReadPort(Port):

  def read(self, block=True):
    assert False, "Unimplemented"


class ReadWritePort(Port):

  def write(self, msg):
    assert self.write_type is not None, "Expected non-None write_type"
    if not self.write_type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.write_type}'")
    assert False, "Unimplemented"

  def read(self, block=True):
    assert False, "Unimplemented"
