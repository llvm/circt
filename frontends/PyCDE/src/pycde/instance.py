#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Union

from numpy import isin

from .appid import AppID

from circt.dialects import hw, msft

import mlir.ir as _ir


class Instance:
  """Represents a _specific_ instance, unique in a design. This is in contrast
  to a module instantiation within another module."""
  import pycde.module as module

  __slots__ = ["parent", "_ref", "_root", "_child_cache", "_spec_mod"]

  global_ref_counter = 0

  @staticmethod
  def _get(root: RootInstance,
           parent: Instance,
           spec_mod: module._SpecializedModule = None):
    self = Instance()
    self._root = root
    self.parent = parent
    self._child_cache = None
    self._spec_mod = spec_mod
    return self

  @property
  def path(self) -> list[Instance]:
    return self.parent.path + [self]

  @property
  def root_instance(self) -> Instance:
    return self._root

  @property
  def _dyn_inst(self) -> msft.DynamicInstanceOp:
    """Return the raw CIRCT op backing this Instance.
       DANGEROUS! If used, take care to not hold on to the result object."""
    return self._root._create_or_get_dyn_inst(self)

  @property
  def _module_symbol(self):
    assert self._spec_mod is not None
    return self._root._system._get_module_symbol(self._spec_mod)

  def _path_attr(self) -> _ir.ArrayAttr:
    module_names = [self._root._module_symbol] + \
        [instance._module_symbol for instance in self.path[:-1]]
    modules = [_ir.StringAttr.get(name) for name in module_names]
    instances = [instance._name_attr for instance in self.path]
    inner_refs = [hw.InnerRefAttr.get(m, i) for m, i in zip(modules, instances)]
    return _ir.ArrayAttr.get(inner_refs)

  @property
  def name(self):
    return str(self._name_attr).strip('"')

  @property
  def _name_attr(self):
    return self._root._get_static_op(self).attributes["sym_name"]

  @property
  def appid(self):
    return AppID(*[i.name for i in self.path])

  def __repr__(self):
    path_names = map(lambda i: i.name, self.path)
    return "<instance: [" + ", ".join(path_names) + "]>"

  def children(self):
    if self._child_cache is not None:
      return self._child_cache
    if self._spec_mod is None:
      return []
    symbols_in_mod = self._root._get_sym_ops_in_module(self._spec_mod)
    children = [self._root._create_instance(self, op) for op in symbols_in_mod]
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
      self._root.placedb.place(self, attr[1], attr[0])
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
    self._root.placedb.place(self, loc, subpath)


class RootInstance(Instance):
  """
  A root of an instance hierarchy starting at top-level 'module'.

  Provides:
    - The placement database.
    - A (necessary) level of indirection into CIRCT IR.
  """
  import pycde.system as system
  import pycde.devicedb as devdb
  from .module import _SpecializedModule

  __slots__ = [
      "_module", "_placedb", "_subsymbol_cache", "_inst_to_static_op_cache",
      "_inst_to_dyn_op_cache", "_system"
  ]

  # TODO: Support rebuilding the caches.

  @staticmethod
  def _get(module: _SpecializedModule, sys: system.System):
    self = RootInstance()
    self._spec_mod = module
    self._system = sys
    self._placedb = None
    self._root = self
    self._subsymbol_cache = {}
    self._inst_to_static_op_cache = {self: sys._get_circt_mod(module)}
    self._inst_to_dyn_op_cache = {}
    self._child_cache = None
    return self

  def _clear_cache(self):
    """Clear out all of the Operation* references."""
    self._subsymbol_cache = None
    self._inst_to_static_op_cache = None
    self._inst_to_dyn_op_cache = None
    self._child_cache = None

  def createdb(self, primdb: devdb.PrimitiveDB = None):
    import pycde.devicedb as devdb
    self._placedb = devdb.PlacementDB(
        self._system._get_circt_mod(self._spec_mod), primdb)

  def _get_static_op(self, inst: Instance):
    # We don't support cache rebuilds yet.
    assert self._inst_to_static_op_cache is not None
    assert inst in self._inst_to_static_op_cache
    return self._inst_to_static_op_cache[inst]

  @property
  def name(self):
    return "<<root>>"

  @property
  def _name_attr(self):
    return None

  @property
  def placedb(self):
    if self._placedb is None:
      raise Exception("Must `createdb` first")
    return self._placedb

  @property
  def path(self) -> list[Instance]:
    return []

  def _create_or_get_dyn_inst(self, inst: Instance):
    # We don't support cache rebuilding yet
    assert self._inst_to_dyn_op_cache is not None
    if inst not in self._inst_to_dyn_op_cache:
      with self._system._get_ip():
        self._inst_to_dyn_op_cache[inst] = \
            msft.DynamicInstanceOp.create(inst._path_attr())
    return self._inst_to_dyn_op_cache[inst]

  def _create_instance(self, parent: Instance, static_op: _ir.Operation):
    import circt.dialects.msft as circtms
    spec_mod = None
    if isinstance(static_op, circtms.InstanceOp):
      spec_mod = self._system._get_symbol_module(static_op.moduleName)
    inst = Instance._get(self, parent, spec_mod)
    self._inst_to_static_op_cache[inst] = static_op
    return inst

  def _get_sym_ops_in_module(self, instance_module: _SpecializedModule):
    if instance_module not in self._subsymbol_cache:
      circt_mod = self._system._get_circt_mod(instance_module)
      if isinstance(circt_mod, msft.MSFTModuleExternOp):
        return []

      def has_symbol(op):
        return "sym_name" in op.attributes

      self._subsymbol_cache[instance_module] = \
          [op for op in circt_mod.entry_block if has_symbol(op)]

    return self._subsymbol_cache[instance_module]
