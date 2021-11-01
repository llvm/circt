#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import typing

from circt import msft

PrimitiveType = msft.PrimitiveType


class PhysLocation:
  __slots__ = ["_loc"]

  def __init__(self,
               prim_type: typing.Union[str, PrimitiveType],
               x: int,
               y: int,
               num: typing.Union[int, None] = None):

    if isinstance(prim_type, str):
      prim_type = getattr(PrimitiveType, prim_type)
    # TODO: Once we get into non-zero num primitives, this needs to be updated.
    if num is None:
      num = 0

    assert isinstance(prim_type, PrimitiveType)
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert isinstance(num, int)
    self._loc = msft.PhysLocationAttr.get(prim_type, x, y, num)

  def __str__(self) -> str:
    loc = self._loc
    return f"PhysLocation<{loc.devtype}, x:{loc.x}, y:{loc.y}, num:{loc.num}>"


class PrimitiveDB:
  __slots__ = ["_db"]

  def __init__(self):
    self._db = msft.PrimitiveDB()

  def add_coords(self,
                 prim_type: typing.Union[str, PrimitiveType],
                 x: int,
                 y: int,
                 num: typing.Union[int, None] = None):
    self.add(PhysLocation(prim_type, x, y, num))

  def add(self, physloc: PhysLocation):
    self._db.add_primitive(physloc._loc)


class PlacementDB:
  __slots__ = ["_db"]

  def __init__(self, _circt_mod, seed: typing.Union[PrimitiveDB, None]):
    self._db = msft.PlacementDB(_circt_mod, seed._db if seed else None)
    self._db.add_design_placements()

  def get_instance_at_coords(self,
                             prim_type: typing.Union[str, PrimitiveType],
                             x: int,
                             y: int,
                             num: typing.Union[int, None] = None) -> object:
    return self.get_instance_at(PhysLocation(prim_type, x, y, num))

  def get_instance_at(self, loc: PhysLocation) -> object:
    inst = self._db.get_instance_at(loc._loc)
    if inst is None:
      return None
    # TODO: resolve instance and return it.
    return inst
