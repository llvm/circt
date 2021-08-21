#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Union
from .appid import AppID

import circt.dialects.hw as hw
from circt import msft

import mlir.ir as ir


class Instance:
  """Represents a _specific_ instance, unique in a design. This is in contrast
  to a module instantiation within another module."""
  import pycde.system as system

  def __init__(self, module: ir.Operation, instOp: hw.InstanceOp,
               parent: Instance, sys: system.System):
    assert module is not None
    self.module = module
    self.instOp = instOp
    self.parent = parent
    assert isinstance(sys, Instance.system.System)
    self.sys = sys

  @property
  def modname(self) -> str:
    modname: str = ir.StringAttr(self.module.attributes["sym_name"]).value
    if modname.startswith("pycde.") or modname.startswith("pycde_"):
      return modname[6:]
    return modname

  @property
  def path(self) -> list[Instance]:
    if self.parent is None:
      return []
    return self.parent.path + [self]

  @property
  def pathAttr(self) -> ir.MlirAttribute:
    symrefs = [f"@{i.name}" for i in self.path]
    return ir.Attribute.parse("::".join(symrefs))

  @property
  def pathToAttr(self) -> ir.MlirAttribute:
    symrefs = [f"@{i.name}" for i in self.path]
    if len(symrefs) <= 1:
      return None
    return ir.Attribute.parse("::".join(symrefs[:-1]))

  @property
  def name(self):
    return ir.StringAttr(self.instOp.instanceName).value

  @property
  def is_root(self):
    return self.parent is None

  @property
  def appid(self):
    return AppID([i.name for i in self.path])

  def __repr__(self):
    path_names = map(lambda i: i.name, self.path)
    return "<instance: [" + ", ".join(path_names) + "]>"

  def walk_instances(self, callback):
    if isinstance(self.module, hw.HWModuleExternOp):
      return
    for op in self.module.entry_block:
      if not isinstance(op, hw.InstanceOp):
        continue

      assert "moduleName" in op.attributes
      tgt_modname = ir.FlatSymbolRefAttr(op.attributes["moduleName"]).value
      tgt_mod = self.sys.get_module(tgt_modname)
      if tgt_mod is None:
        continue
      inst = Instance(tgt_mod, op, self, self.sys)
      callback(inst)
      inst.walk_instances(callback)

  def attach_attribute(self, attr_key: str, attr: ir.Attribute):
    # In the case where this instance sits in the 'top' or 'root' module, we
    # don't need a switch attr.
    if self.parent.is_root:
      self.instOp.attributes[attr_key] = attr
      return

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
    cases.append((self.pathToAttr, attr))
    self.instOp.attributes[attr_key] = msft.SwitchInstanceAttr.get(cases)

  def place(self,
            subpath: Union[str, list[str]],
            devtype: msft.DeviceType,
            x: int,
            y: int,
            num: int = 0):
    loc = msft.PhysLocationAttr.get(devtype, x, y, num)
    if isinstance(subpath, list):
      subpath = "|".join(subpath)
    self.attach_attribute(f"loc:{subpath}", loc)
