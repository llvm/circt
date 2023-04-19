#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum

from circt.ir import Attribute, Type

from ._firrtl_ops_gen import *


class Direction(Enum):
  IN = 0
  OUT = 1


@dataclass
class DirectionType:
  direction: Direction
  type: Type


@dataclass
class Port:
  name: str
  direction: Direction
  type: Type


def Input(type: Type) -> DirectionType:
  return DirectionType(direction=Direction.IN, type=type)


def Output(type: Type) -> DirectionType:
  return DirectionType(direction=Direction.OUT, type=type)


def UInt(bitwidth: int) -> Type:
  return Type.parse(f"!firrtl.uint<{bitwidth}>")


def SInt(bitwidth: int) -> Type:
  return Type.parse(f"!firrtl.sint<{bitwidth}>")


def Clock() -> Type:
  return Type.parse(f"!firrtl.clock")


def Vec(size: int, inner: Type) -> Type:
  return Type.parse(f"!firrtl.vector<{str(inner)[8:]}, {size}>")


def Convention(conv: str) -> Attribute:
  return Attribute.parse(f"#firrtl<convention {conv}>")


def NameKind(kind: str) -> Attribute:
  return Attribute.parse(f"#firrtl<name_kind {kind}>")
