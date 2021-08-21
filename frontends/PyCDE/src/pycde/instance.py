#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import circt.dialects.hw as hw

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
  def path(self) -> list[Instance]:
    if self.parent is None:
      return []
    return self.parent.path + [self]

  @property
  def name(self):
    return ir.StringAttr(self.instOp.instanceName).value

  def __repr__(self):
    path_names = map(lambda i: i.name, self.path)
    return "<instance: [" + ", ".join(path_names) + "]>"

  def walk_instances(self, callback):
    for op in self.module.entry_block:
      if not isinstance(op, hw.InstanceOp):
        continue

      assert "moduleName" in op.attributes
      tgt_modname = ir.FlatSymbolRefAttr(op.attributes["moduleName"]).value
      tgt_mod = self.sys.get_module(tgt_modname)
      if tgt_mod is None:
        raise ValueError(f"Could not find module {tgt_modname}")
      inst = Instance(tgt_mod, op, self, self.sys)
      callback(inst)
      inst.walk_instances(callback)
