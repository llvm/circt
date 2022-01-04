#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import typing

from circt.dialects import msft

from mlir.ir import StringAttr, ArrayAttr

PrimitiveType = msft.PrimitiveType


class PhysLocation:
  __slots__ = ["_loc"]

  def __init__(self,
               prim_type: typing.Union[str, PrimitiveType],
               x: int,
               y: int,
               num: typing.Union[int, None] = None,
               sub_path: str = ""):

    if isinstance(prim_type, str):
      prim_type = getattr(PrimitiveType, prim_type)
    # TODO: Once we get into non-zero num primitives, this needs to be updated.
    if num is None:
      num = 0

    assert isinstance(prim_type, PrimitiveType)
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert isinstance(num, int)
    self._loc = msft.PhysLocationAttr.get(prim_type, x, y, num, sub_path)

  def __str__(self) -> str:
    loc = self._loc
    return f"PhysLocation<{loc.devtype}, x:{loc.x}, y:{loc.y}, num:{loc.num}>"


class PhysicalRegion:
  _counter = 0
  _used_names = set([])

  __slots__ = ["_physical_region"]

  def __init__(self, name: str = None, bounds: list = None):
    if name is None or name in PhysicalRegion._used_names:
      prefix = name if name is not None else "region"
      name = f"{prefix}_{PhysicalRegion._counter}"
      while name in PhysicalRegion._used_names:
        PhysicalRegion._counter += 1
        name = f"{prefix}_{PhysicalRegion._counter}"
    PhysicalRegion._used_names.add(name)

    if bounds is None:
      bounds = []

    name_attr = StringAttr.get(name)
    bounds_attr = ArrayAttr.get(bounds)
    self._physical_region = msft.PhysicalRegionOp(name_attr, bounds_attr)

  def add_bounds(self, x_bounds: tuple, y_bounds: tuple):
    """Add a new bounding box to the region."""
    if (len(x_bounds) != 2):
      raise ValueError(f"expected lower and upper x bounds, got: {x_bounds}")
    if (len(y_bounds) != 2):
      raise ValueError(f"expected lower and upper y bounds, got: {y_bounds}")

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    bounds = msft.PhysicalBoundsAttr.get(x_min, x_max, y_min, y_max)

    self._physical_region.add_bounds(bounds)

    return self

  def get_ref(self):
    """Get a pair suitable for add_attribute to attach to an operation."""
    name = self._physical_region.sym_name.value
    return ("loc", msft.PhysicalRegionRefAttr.get(name))


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
