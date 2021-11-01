#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Union

from pycde.devicedb import PhysLocation, PrimitiveDB, PlacementDB
from .appid import AppID

import circt.dialects.hw as hw
from circt import msft

import mlir.ir as ir


# TODO: bug: holds an Operation* without releasing it. Use a level of
# indirection.
class Instance:
  """Represents a _specific_ instance, unique in a design. This is in contrast
  to a module instantiation within another module."""
  import pycde.system as system

  def __init__(self,
               module: type,
               instOp: hw.InstanceOp,
               parent: Instance,
               sys: system.System,
               primdb: PrimitiveDB = None):
    assert module is not None
    self.module = module
    self.instOp = instOp
    self.parent = parent
    if parent is None:
      self.placedb = PlacementDB(sys._get_circt_mod(module), primdb)
    assert isinstance(sys, Instance.system.System)
    self.sys = sys

  @property
  def path(self) -> list[Instance]:
    if self.parent is None:
      return []
    return self.parent.path + [self]

  @property
  def root_module(self) -> hw.HWModuleOp:
    if self.parent is None:
      return self.module
    return self.parent.root_module

  @property
  def root_instance(self) -> Instance:
    if self.parent is None:
      return self
    return self.parent.root_instance

  @property
  def path_attr(self) -> msft.RootedInstancePathAttr:
    return msft.RootedInstancePathAttr.get(
        ir.FlatSymbolRefAttr.get(self.sys._get_module_symbol(self.root_module)),
        [x.name_attr for x in self.path[:-1]])

  @property
  def name(self):
    return ir.StringAttr(self.instOp.instanceName).value

  @property
  def name_attr(self):
    return ir.StringAttr(self.instOp.instanceName)

  @property
  def is_root(self):
    return self.parent is None

  @property
  def appid(self):
    return AppID(*[i.name for i in self.path])

  def __repr__(self):
    path_names = map(lambda i: i.name, self.path)
    return "<instance: [" + ", ".join(path_names) + "]>"

  def walk(self, callback):
    """Descend the instance hierarchy, calling back on each instance."""
    circt_mod = self.sys._get_circt_mod(self.module)
    if isinstance(circt_mod, hw.HWModuleExternOp):
      return
    for op in circt_mod.entry_block:
      if not isinstance(op, hw.InstanceOp):
        continue

      assert "moduleName" in op.attributes
      tgt_modname = ir.FlatSymbolRefAttr(op.attributes["moduleName"]).value
      tgt_mod = self.sys._get_symbol_module(tgt_modname).modcls
      assert tgt_mod is not None
      inst = Instance(tgt_mod, op, self, self.sys)
      callback(inst)
      inst.walk(callback)

  def _attach_attribute(self, attr_key: str, attr: ir.Attribute):
    if isinstance(attr, PhysLocation):
      assert attr_key.startswith("loc:")
      db = self.root_instance.placedb._db
      attr = attr._loc
      rc = db.add_placement(attr, self.path_attr, attr_key[4:],
                            self.instOp.operation)
      if not rc:
        raise ValueError("Failed to place")

    if attr_key not in self.instOp.attributes:
      cases = []
    else:
      existing_attr = self.instOp.attributes[attr_key]
      try:
        inst_switch = msft.SwitchInstanceAttr(existing_attr)
        cases = inst_switch.cases
      except TypeError:
        raise ValueError(
            f"Existing attribute ({existing_attr}) is not msft.switch.inst.")
    cases.append((self.path_attr, attr))
    self.instOp.attributes[attr_key] = msft.SwitchInstanceAttr.get(cases)

  def place(self,
            subpath: Union[str, list[str]],
            devtype: msft.PrimitiveType,
            x: int,
            y: int,
            num: int = 0):
    loc = msft.PhysLocationAttr.get(devtype, x, y, num)
    if isinstance(subpath, list):
      subpath = "|".join(subpath)
    self._attach_attribute(f"loc:{subpath}", loc)
