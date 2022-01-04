#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Union

from pycde.devicedb import PhysLocation, PrimitiveDB, PlacementDB
from .appid import AppID

from circt.dialects import hw, msft, seq

import mlir.ir as ir


# TODO: bug: holds an Operation* without releasing it. Use a level of
# indirection.
class Instance:
  """Represents a _specific_ instance, unique in a design. This is in contrast
  to a module instantiation within another module."""
  import pycde.system as system

  global_ref_counter = 0

  def __init__(self,
               module: type,
               instOp: Union[msft.InstanceOp, seq.CompRegOp],
               parent: Instance,
               sys: system.System,
               primdb: PrimitiveDB = None):
    assert module is not None
    assert instOp is None or (isinstance(instOp, msft.InstanceOp) or
                              isinstance(instOp, seq.CompRegOp))
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
  def path_attr(self) -> ir.ArrayAttr:
    module_names = [self.sys._get_module_symbol(self.root_module)] + [
        self.sys._get_module_symbol(instance.module)
        for instance in self.path[:-1]
    ]
    modules = [ir.StringAttr.get(name) for name in module_names]
    instances = [instance.name_attr for instance in self.path]
    inner_refs = [hw.InnerRefAttr.get(m, i) for m, i in zip(modules, instances)]
    return ir.ArrayAttr.get(inner_refs)

  @property
  def name(self):
    return self.name_attr.value

  @property
  def name_attr(self):
    if isinstance(self.instOp, msft.InstanceOp):
      return ir.StringAttr(self.instOp.sym_name)
    elif isinstance(self.instOp, seq.CompRegOp):
      return ir.StringAttr(self.instOp.innerSym)

  @property
  def is_root(self):
    return self.parent is None

  @property
  def appid(self):
    return AppID(*[i.name for i in self.path])

  @classmethod
  def get_global_ref_symbol(cls):
    counter = cls.global_ref_counter
    cls.global_ref_counter += 1
    return ir.StringAttr.get("ref" + str(counter))

  def __repr__(self):
    path_names = map(lambda i: i.name, self.path)
    return "<instance: [" + ", ".join(path_names) + "]>"

  def walk(self, callback):
    """Descend the instance hierarchy, calling back on each instance."""
    circt_mod = self.sys._get_circt_mod(self.module)
    if isinstance(circt_mod, msft.MSFTModuleExternOp):
      return
    for op in circt_mod.entry_block:
      if isinstance(op, seq.CompRegOp):
        inst = Instance(circt_mod, op, self, self.sys)
        callback(inst)
        continue

      if not isinstance(op, msft.InstanceOp):
        continue

      assert "moduleName" in op.attributes
      tgt_modname = ir.FlatSymbolRefAttr(op.attributes["moduleName"]).value
      tgt_mod = self.sys._get_symbol_module(tgt_modname).modcls
      assert tgt_mod is not None
      inst = Instance(tgt_mod, op, self, self.sys)
      callback(inst)
      inst.walk(callback)

  def _attach_attribute(self, sub_path: str, attr: ir.Attribute):
    if isinstance(attr, PhysLocation):
      attr = attr._loc

    db = self.root_instance.placedb._db
    rc = db.add_placement(attr, self.path_attr, sub_path, self.instOp.operation)
    if not rc:
      raise ValueError("Failed to place")

    # Create a global ref to this path.
    global_ref_symbol = Instance.get_global_ref_symbol()
    path_attr = self.path_attr
    with ir.InsertionPoint(self.sys.mod.body):
      global_ref = hw.GlobalRefOp(global_ref_symbol, path_attr)

    # Attach the attribute to the global ref.
    global_ref.attributes["loc:" + sub_path] = attr

    # Add references to the global ref for each instance through the hierarchy.
    for instance in self.path:
      # Find any existing global refs.
      if "circt.globalRef" in instance.instOp.attributes:
        global_refs = [
            ref for ref in ir.ArrayAttr(
                instance.instOp.attributes["circt.globalRef"])
        ]
      else:
        global_refs = []

      # Add the new global ref.
      global_refs.append(hw.GlobalRefAttr.get(global_ref_symbol))
      global_refs_attr = ir.ArrayAttr.get(global_refs)
      instance.instOp.attributes["circt.globalRef"] = global_refs_attr

      # Set the expected inner_sym attribute on the instance to abide by the
      # global ref contract.
      instance.instOp.attributes["inner_sym"] = instance.name_attr

  def place(self,
            subpath: Union[str, list[str]],
            devtype: msft.PrimitiveType,
            x: int,
            y: int,
            num: int = 0):
    if isinstance(subpath, list):
      subpath = "|".join(subpath)
    loc = msft.PhysLocationAttr.get(devtype, x, y, num, subpath)
    self._attach_attribute(subpath, loc)
