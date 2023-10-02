#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Type

from . import hw, msft as _msft
from . import _hw_ops_ext as _hw_ext
from .. import support

from .. import ir as _ir


class PhysicalRegionOp:

  def add_bounds(self, bounds):
    existing_bounds = [b for b in _ir.ArrayAttr(self.attributes["bounds"])]
    existing_bounds.append(bounds)
    new_bounds = _ir.ArrayAttr.get(existing_bounds)
    self.attributes["bounds"] = new_bounds


class InstanceOp:

  @property
  def moduleName(self):
    return _ir.FlatSymbolRefAttr(self.attributes["moduleName"])


class EntityExternOp:

  @staticmethod
  def create(symbol, metadata=""):
    symbol_attr = support.var_to_attribute(symbol)
    metadata_attr = support.var_to_attribute(metadata)
    return _msft.EntityExternOp(symbol_attr, metadata_attr)


class InstanceHierarchyOp:

  @staticmethod
  def create(root_mod, instance_name=None):
    hier = _msft.InstanceHierarchyOp(root_mod, instName=instance_name)
    hier.body.blocks.append()
    return hier

  @property
  def top_module_ref(self):
    return self.attributes["topModuleRef"]


class DynamicInstanceOp:

  @staticmethod
  def create(name_ref):
    inst = _msft.DynamicInstanceOp(name_ref)
    inst.body.blocks.append()
    return inst

  @property
  def instance_path(self):
    path = []
    next = self
    while isinstance(next, DynamicInstanceOp):
      path.append(next.attributes["instanceRef"])
      next = next.operation.parent.opview
    path.reverse()
    return _ir.ArrayAttr.get(path)

  @property
  def instanceRef(self):
    return self.attributes["instanceRef"]


class PDPhysLocationOp:

  @property
  def loc(self):
    return _msft.PhysLocationAttr(self.attributes["loc"])
