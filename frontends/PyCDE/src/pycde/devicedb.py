#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import typing

from circt.dialects import hw, msft

from mlir.ir import StringAttr, ArrayAttr, FlatSymbolRefAttr

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
  __slots__ = ["_db", "_circt_mod"]

  def __init__(self, _circt_mod, seed: typing.Union[PrimitiveDB, None]):
    self._db = msft.PlacementDB(_circt_mod, seed._db if seed else None)
    self._db.add_design_placements()
    self._circt_mod = _circt_mod

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

  def reserve_location(self, loc: PhysLocation, entity: EntityExtern):
    sym_name = entity._entity_extern.sym_name.value
    ref = FlatSymbolRefAttr.get(sym_name)
    path = ArrayAttr.get([ref])
    subpath = ""
    self._db.add_placement(loc._loc, path, subpath, entity._entity_extern)

  def remove_placement(self, loc: PhysLocation):
    success = self._db.remove_placement(loc._loc)
    if not success:
      raise RuntimeError(f"Unable to remove placement at: {loc}")

    self._mutate_global_ref_placement(loc)

  def move_placement(self, old_loc: PhysLocation, new_loc: PhysLocation):
    success = self._db.move_placement(old_loc._loc, new_loc._loc)
    if not success:
      raise RuntimeError(
          f"Unable to move placement from: {old_loc._loc}, to: {new_loc._loc}")

    self._mutate_global_ref_placement(old_loc, new_loc)

  # Mutate the IR to reflect a placement that is being removed or moved. The old
  # location is used to scan for the relevant global ref. If the new location is
  # None, the old location is removed. Otherwise, the new location replaces the
  # old location. It would be best to accept an AppID or some other pointer to
  # indicate which op needs to have its global ref updated, but we currently
  # have no way to find an entity without walking the IR, so we just scan for
  # the relevant global ref.
  def _mutate_global_ref_placement(self,
                                   old_loc: PhysLocation,
                                   new_loc: PhysLocation = None):
    # Find the top-level module.
    top_mod = self._circt_mod.operation.parent
    assert top_mod.parent is None

    # Find the global ref in question.
    global_ref_to_remove = None
    for op in top_mod.regions[0].blocks[0]:
      # If we already found it break.
      if global_ref_to_remove:
        break
      if isinstance(op, hw.GlobalRefOp):
        for nvp in op.attributes:
          if nvp.attr == old_loc._loc:
            # If the global ref specified this location, remove or update it.
            if new_loc is None:
              del op.attributes[nvp.name]
            else:
              op.attributes[nvp.name] = new_loc._loc

            # If the global ref only has its inherent attributes, erase it.
            if len(op.attributes) == 2:
              global_ref_to_remove = op

    # Erase the global ref if necessary. Note that this leaves a dangling
    # reference in the IR. This is not ideal, but as mentioned above it would
    # currently require a full walk of the IR to find and remove the reference.
    # The reference is ultimately dropped regardless when MSFT lowers to HW.
    if global_ref_to_remove:
      global_ref_to_remove.operation.erase()


class EntityExtern:
  __slots__ = ["_entity_extern"]

  def __init__(self, tag: str, metadata: typing.Any = ""):
    self._entity_extern = msft.EntityExternOp.create(tag, metadata)
