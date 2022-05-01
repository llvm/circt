#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

from .appid import AppID

from circt.dialects import msft

import mlir.ir as ir


class InstanceLike:
  """Parent class for anything which should be walked and can contain PD ops."""
  from .module import _SpecializedModule

  __slots__ = ["inside_of", "tgt_mod", "root", "_child_cache", "_op_cache"]

  def __init__(self, inside_of: _SpecializedModule,
               tgt_mod: Optional[_SpecializedModule], root: InstanceHierarchy):
    """
    Construct a new instance. Since the terminology can be confusing:
    - inside_of: the module which contains this instance (e.g. the instantiation
      site).
    - tgt_mod: if the instance is an instantation, `tgt_mod` is the module being
      instantiated. Examples of things which aren't instantiations:
      `seq.compreg`s.
    """

    self.inside_of = inside_of
    self.tgt_mod = tgt_mod
    self.root = root
    self._child_cache: Dict[ir.StringAttr, Instance] = None
    self._op_cache = root.system._op_cache

  def _create_instance(self, parent: Instance,
                       static_op: ir.Operation) -> Instance:
    """Create a new `Instance` which is a child of `parent` in the instance
    hierarchy and corresponds to the given static operation. The static
    operation need not be a module instantiation."""

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

  def _get_ip(self) -> ir.InsertionPoint:
    return ir.InsertionPoint(self._dyn_inst.body.blocks[0])

  @property
  def _inside_of_symbol(self) -> str:
    """Return the string symbol of the module which contains this instance."""
    return self._op_cache.get_module_symbol(self.inside_of)

  def __repr__(self) -> str:
    path_names = [i.name for i in self.path]
    return "<instance: [" + ", ".join(path_names) + "]>"

  @property
  def appid(self) -> AppID:
    return AppID(*[i.name for i in self.path])

  def children(self) -> Dict[str, Instance]:
    """Return a list of this instances children. Cache said list."""
    if self._child_cache is not None:
      return self._child_cache
    symbols_in_mod = self._op_cache.get_sym_ops_in_module(self.tgt_mod)
    children = {
        sym: self._create_instance(self, op)
        for (sym, op) in symbols_in_mod.items()
    }
    # TODO: make these weak refs
    self._child_cache = children
    return children

  def walk(self, callback):
    """Descend the instance hierarchy, calling back on each instance."""
    callback(self)
    for child in self.children().values():
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

  @property
  def locations(self) -> List[Tuple[object, str]]:
    """Returns a list of physical locations assigned to this instance in
    (PhysLocation, subpath) format."""

    def conv(op):
      import pycde.devicedb as devdb
      loc = devdb.PhysLocation(op.loc)
      subPath = op.subPath
      if subPath is not None:
        subPath = ir.StringAttr(subPath).value
      return (loc, subPath)

    dyn_inst_block = self._dyn_inst.operation.regions[0].blocks[0]
    return [
        conv(op)
        for op in dyn_inst_block
        if isinstance(op, msft.PDPhysLocationOp)
    ]


class Instance(InstanceLike):
  """Represents a _specific_ instance, unique in a design. This is in contrast
  to a module instantiation within another module."""

  from .module import _SpecializedModule

  __slots__ = ["parent", "symbol"]

  def __init__(self, parent: Instance, instance_sym: ir.Attribute,
               inside_of: _SpecializedModule,
               tgt_mod: Optional[_SpecializedModule], root: InstanceHierarchy):
    super().__init__(inside_of, tgt_mod, root)
    self.parent = parent
    self.symbol = instance_sym

  @property
  def _dyn_inst(self) -> msft.DynamicInstanceOp:
    """Returns the raw CIRCT op backing this Instance."""
    op = self._op_cache.create_or_get_dyn_inst(self)
    if op is None:
      raise InstanceDoesNotExistError(str(self))
    return op

  @property
  def path(self) -> list[Instance]:
    return self.parent.path + [self]

  @property
  def name(self) -> str:
    return ir.StringAttr(self.symbol).value


class InstanceHierarchy(InstanceLike):
  """
  A root of an instance hierarchy starting at top-level 'module'. Different from
  an `Instance` since the base cases differ, and doesn't have an instance symbol
  (since it addresses the 'top' module). Plus, CIRCT models it this way.
  """
  import pycde.system as cdesys
  from .module import _SpecializedModule

  __slots__ = ["system"]

  def __init__(self, module: _SpecializedModule, sys: cdesys.System):
    self.system = sys
    super().__init__(inside_of=module, tgt_mod=module, root=self)
    sys._op_cache.create_instance_hier_op(self)

  @property
  def _dyn_inst(self) -> msft.InstanceHierarchyOp:
    """Returns the raw CIRCT op backing this Instance."""
    op = self._op_cache.get_instance_hier_op(self)
    if op is None:
      raise InstanceDoesNotExistError(self.inside_of.modcls.__name__)
    return op

  @property
  def name(self) -> str:
    return "<<root>>"

  @property
  def path(self) -> list:
    return []


class InstanceError(Exception):
  """An error related to dynamic instances."""

  def __init__(self, msg: str):
    super().__init__(msg)


class InstanceDoesNotExistError(Exception):
  """The instance which you are trying to reach does not exist (anymore)"""

  def __init__(self, inst_str: str):
    super().__init__(f"Instance {self} does not exist (anymore)")
