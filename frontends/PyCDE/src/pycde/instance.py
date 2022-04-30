#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import List, Optional, Union

from .appid import AppID

from circt.dialects import hw, msft

import mlir.ir as _ir


class InstanceLike:
  from .module import _SpecializedModule

  def __init__(self, inside_of: _SpecializedModule,
               tgt_mod: Optional[_SpecializedModule], root: InstanceHierarchy):
    self.inside_of = inside_of
    self.tgt_mod = tgt_mod
    self.root = root
    self._child_cache: List[Instance] = None
    self._op_cache = root.system._op_cache

  def _create_instance(self, parent: Instance, static_op: _ir.Operation):
    sym_name = static_op.attributes["sym_name"]
    tgt_mod = None
    if isinstance(static_op, msft.InstanceOp):
      tgt_mod = self._op_cache.get_symbol_module(static_op.moduleName)
    inst = Instance(parent,
                    instance_sym=sym_name,
                    inside_of=self.tgt_mod,
                    tgt_mod=tgt_mod,
                    root=self.root)
    return inst

  def _get_sym_ops_in_module(self):
    if self.tgt_mod is None:
      return []
    circt_mod = self._op_cache.get_circt_mod(self.tgt_mod)
    if isinstance(circt_mod, msft.MSFTModuleExternOp):
      return []

    def has_symbol(op):
      return "sym_name" in op.attributes

    return [op for op in circt_mod.entry_block if has_symbol(op)]

  @property
  def _dyn_inst(self) -> msft.DynamicInstanceOp:
    """Return the raw CIRCT op backing this Instance.
       DANGEROUS! If used, take care to not hold on to the result object."""
    return self._op_cache._create_or_get_dyn_inst(self)

  @property
  def _inside_of_symbol(self):
    return self._op_cache.get_module_symbol(self.inside_of)

  def __repr__(self):
    path_names = map(lambda i: i.name, self.path)
    return "<instance: [" + ", ".join(path_names) + "]>"

  @property
  def appid(self):
    return AppID(*[i.name for i in self.path])

  def children(self):
    if self._child_cache is not None:
      return self._child_cache
    symbols_in_mod = self._get_sym_ops_in_module()
    children = [self._create_instance(self, op) for op in symbols_in_mod]
    # TODO: make these weak refs
    self._child_cache = children
    return children

  def walk(self, callback):
    """Descend the instance hierarchy, calling back on each instance."""
    callback(self)
    for child in self.children():
      child.walk(callback)

  def _attach_attribute(self, attr):
    import pycde.devicedb as devdb

    assert isinstance(attr, tuple), "Only (subpath, loc) are supported"
    if isinstance(attr[1], devdb.PhysLocation):
      self.root.system.placedb.place(self, attr[1], attr[0])
    else:
      assert False

  def place(self,
            devtype: msft.PrimitiveType,
            x: int,
            y: int,
            num: int = 0,
            subpath: Union[str, list[str]] = ""):
    import pycde.devicedb as devdb
    if isinstance(subpath, list):
      subpath = "|".join(subpath)
    loc = devdb.PhysLocation(devtype, x, y, num)
    self.root.system.placedb.place(self, loc, subpath)


class Instance(InstanceLike):
  """Represents a _specific_ instance, unique in a design. This is in contrast
  to a module instantiation within another module."""

  from .module import _SpecializedModule

  __slots__ = ["parent", "_ref", "module"]

  def __init__(self, parent: Instance, instance_sym: _ir.Attribute,
               inside_of: _SpecializedModule,
               tgt_mod: Optional[_SpecializedModule], root: InstanceHierarchy):
    super().__init__(inside_of, tgt_mod, root)
    self.parent = parent
    self.symbol = instance_sym
    self._ref = hw.InnerRefAttr.get(_ir.StringAttr.get(self._inside_of_symbol),
                                    instance_sym)

  @property
  def path(self) -> list[Instance]:
    return self.parent.path + [self]

  def _get_ip(self) -> _ir.InsertionPoint:
    return _ir.InsertionPoint(self._dyn_inst.body.blocks[0])

  @property
  def name(self):
    return _ir.StringAttr(self.symbol).value


class InstanceHierarchy(InstanceLike):
  """
  A root of an instance hierarchy starting at top-level 'module'.

  Provides a (necessary) level of indirection into CIRCT IR.
  """
  import pycde.system as system
  from .module import _SpecializedModule

  def __init__(self, module: _SpecializedModule, sys: system.System):
    self.system = sys
    super().__init__(inside_of=module, tgt_mod=module, root=self)
    sys._op_cache.get_or_create_instance_hier_op(self)

  @property
  def name(self):
    return "<<root>>"

  def _get_ip(self) -> _ir.InsertionPoint:
    return _ir.InsertionPoint(
        self._op_cache.get_or_create_instance_hier_op(self).instances.blocks[0])

  @property
  def path(self):
    return []
