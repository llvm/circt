#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .circt.dialects import esi
from .circt import ir

from .types import Type, Bundle, Channel, ChannelSignaling, ClockType, Bits

from functools import singledispatchmethod
from typing import Callable, Optional


class ModuleDecl(property):
  """Represents an input or output port on a design module."""

  __slots__ = ["idx", "name", "type"]

  def __init__(self,
               type: Type,
               name: Optional[str] = None,
               fget: Optional[Callable] = None,
               fset: Optional[Callable] = None):
    super().__init__(fget=fget, fset=fset)
    self.idx: Optional[int] = None
    self.name = name
    self.type = type


class Output(ModuleDecl):
  """Create an RTL-level output port"""

  def __init__(self, type: Type, name: Optional[str] = None):
    from .signals import _FromCirctValue

    def fget(mod_inst, self=self):
      return _FromCirctValue(mod_inst.inst.operation.results[self.idx])

    super().__init__(type, name, fget=fget)

  def __repr__(self) -> str:
    return f"output '{self.name}': {self.type}"


class OutputChannel(Output):
  """Create an ESI output channel port."""

  def __init__(self,
               type: Type,
               signaling: int = ChannelSignaling.ValidReady,
               name: Optional[str] = None):
    type = Channel(type, signaling)
    super().__init__(type, name)


class SendBundle(Output):
  """Create an ESI bundle output port (aka sending port)."""

  def __init__(self, bundle: Bundle, name: Optional[str] = None):
    super().__init__(bundle, name)


class Input(ModuleDecl):
  """Create an RTL-level input port."""

  def __init__(self, type: Type, name: Optional[str] = None):
    from .signals import _FromCirctValue

    def fget(mod_inst, self=self):
      return _FromCirctValue(mod_inst.inst.operation.operands[self.idx])

    super().__init__(type, name, fget=fget)

  def __repr__(self) -> str:
    return f"input '{self.name}': {self.type}"


class Clock(Input):
  """Create a clock input"""

  def __init__(self, name: Optional[str] = None):
    super().__init__(ClockType(), name)

  def __repr__(self) -> str:
    return f"clock {self.name}"


class Reset(Input):
  """Create a reset input."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(Bits(1), name)


class InputChannel(Input):
  """Create an ESI input channel port."""

  def __init__(self,
               type: Type,
               signaling: int = ChannelSignaling.ValidReady,
               name: str = None):
    type = Channel(type, signaling)
    super().__init__(type, name)


class RecvBundle(Input):
  """Create an ESI bundle input port (aka receiving port)."""

  def __init__(self, bundle: Bundle, name: str = None):
    super().__init__(bundle, name)


class AppID:
  AttributeName = "esi.appid"

  @singledispatchmethod
  def __init__(self, name: str, idx: Optional[int] = None):
    self._appid = esi.AppIDAttr.get(name, idx)

  @__init__.register(ir.Attribute)
  def __init__mlir_attr(self, attr: ir.Attribute):
    self._appid = esi.AppIDAttr(attr)

  @property
  def name(self) -> str:
    return self._appid.name

  @property
  def index(self) -> int:
    return self._appid.index

  def __repr__(self) -> str:
    return f"{self.name}[{self.index}]"


class _PyProxy:
  """Parent class for a Python object which has a corresponding IR op (i.e. a
  proxy class)."""

  __slots__ = ["name"]

  def __init__(self, name: str):
    self.name = name

  def clear_op_refs(self):
    """Clear all references to IR ops."""
    pass


class PortError(Exception):
  pass
