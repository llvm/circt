#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import sv, hw
from .. import support
from .._mlir_libs._circt._sv import *
from ..dialects._ods_common import _cext as _ods_cext
from ..ir import ArrayAttr, Attribute, FlatSymbolRefAttr, OpView, StringAttr
from ._sv_ops_gen import *
from ._sv_ops_gen import _Dialect


@_ods_cext.register_operation(_Dialect, replace=True)
class IfDefOp(IfDefOp):

  def __init__(self, cond: Attribute, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {"cond": cond}
    regions = 2
    super().__init__(
        self.build_generic(attributes=attributes,
                           results=results,
                           operands=operands,
                           successors=None,
                           regions=regions,
                           loc=loc,
                           ip=ip))
    self.regions[0].blocks.append()
    self.regions[1].blocks.append()


@_ods_cext.register_operation(_Dialect, replace=True)
class WireOp(WireOp):

  def __init__(self,
               data_type,
               name,
               *,
               sym_name=None,
               svAttributes=None,
               loc=None,
               ip=None):
    attributes = {"name": StringAttr.get(name)}
    if sym_name is not None:
      attributes["inner_sym"] = hw.InnerSymAttr.get(StringAttr.get(sym_name))
    if svAttributes is not None:
      attributes["svAttributes"] = ArrayAttr.get(svAttributes)
    OpView.__init__(
        self,
        self.build_generic(attributes=attributes,
                           results=[data_type],
                           operands=[],
                           successors=None,
                           regions=0,
                           loc=loc,
                           ip=ip))

  @staticmethod
  def create(data_type, name=None, sym_name=None):
    if not isinstance(data_type, hw.InOutType):
      data_type = hw.InOutType.get(data_type)
    return sv.WireOp(data_type, name, sym_name=sym_name)


@_ods_cext.register_operation(_Dialect, replace=True)
class RegOp(RegOp):

  def __init__(self,
               data_type,
               name,
               *,
               sym_name=None,
               svAttributes=None,
               loc=None,
               ip=None):
    attributes = {"name": StringAttr.get(name)}
    if sym_name is not None:
      attributes["inner_sym"] = hw.InnerSymAttr.get(StringAttr.get(sym_name))
    if svAttributes is not None:
      attributes["svAttributes"] = ArrayAttr.get(svAttributes)
    OpView.__init__(
        self,
        self.build_generic(attributes=attributes,
                           results=[data_type],
                           operands=[],
                           successors=None,
                           regions=0,
                           loc=loc,
                           ip=ip))


@_ods_cext.register_operation(_Dialect, replace=True)
class AssignOp(AssignOp):

  @staticmethod
  def create(dest, src):
    return sv.AssignOp(dest=dest, src=src)


@_ods_cext.register_operation(_Dialect, replace=True)
class ReadInOutOp(ReadInOutOp):

  @staticmethod
  def create(value):
    value = support.get_value(value)
    type = support.get_self_or_inner(value.type).element_type
    return sv.ReadInOutOp(value)
