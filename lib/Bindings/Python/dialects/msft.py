#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import hw, msft
from .. import support
from .._mlir_libs._circt._msft import *
from ..dialects._ods_common import _cext as _ods_cext
from ..ir import ArrayAttr
from ._msft_ops_gen import *
from ._msft_ops_gen import _Dialect
from typing import Dict, List, Type


@_ods_cext.register_operation(_Dialect, replace=True)
class DeclPhysicalRegionOp(DeclPhysicalRegionOp):

  def add_bounds(self, bounds):
    existing_bounds = [b for b in ArrayAttr(self.attributes["bounds"])]
    existing_bounds.append(bounds)
    new_bounds = ArrayAttr.get(existing_bounds)
    self.attributes["bounds"] = new_bounds


@_ods_cext.register_operation(_Dialect, replace=True)
class InstanceHierarchyOp(InstanceHierarchyOp):

  @staticmethod
  def create(root_mod, instance_name=None):
    hier = msft.InstanceHierarchyOp(root_mod, instName=instance_name)
    hier.body.blocks.append()
    return hier

  @property
  def top_module_ref(self):
    return self.attributes["topModuleRef"]


@_ods_cext.register_operation(_Dialect, replace=True)
class DynamicInstanceOp(DynamicInstanceOp):

  @staticmethod
  def create(name_ref):
    inst = msft.DynamicInstanceOp(name_ref)
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
    return ArrayAttr.get(path)

  @property
  def instanceRef(self):
    return self.attributes["instanceRef"]


@_ods_cext.register_operation(_Dialect, replace=True)
class PDPhysLocationOp(PDPhysLocationOp):

  @property
  def loc(self):
    return msft.PhysLocationAttr(self.attributes["loc"])
