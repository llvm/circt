#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..dialects._ods_common import _cext as _ods_cext
from ..ir import IntegerAttr, IntegerType
from ..support import NamedValueOpView, get_value
from ._hwarith_ops_gen import *
from ._hwarith_ops_gen import _Dialect


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


def BinaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, lhs=None, rhs=None, result_type=None):
      return cls([get_value(lhs), get_value(rhs)])

  return _Class


@BinaryOp
@_ods_cext.register_operation(_Dialect, replace=True)
class DivOp(DivOp):
  pass


@BinaryOp
@_ods_cext.register_operation(_Dialect, replace=True)
class SubOp(SubOp):
  pass


@BinaryOp
@_ods_cext.register_operation(_Dialect, replace=True)
class AddOp(AddOp):
  pass


@BinaryOp
@_ods_cext.register_operation(_Dialect, replace=True)
class MulOp(MulOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
class CastOp(CastOp):

  @classmethod
  def create(cls, value, result_type):
    return cls(result_type, value)


@_ods_cext.register_operation(_Dialect, replace=True)
class ICmpOp(ICmpOp):
  # Predicate constants.

  # `==` and `!=`
  PRED_EQ = 0b000
  PRED_NE = 0b001
  # `<` and `>=`
  PRED_LT = 0b010
  PRED_GE = 0b011
  # `<=` and `>`
  PRED_LE = 0b100
  PRED_GT = 0b101

  @classmethod
  def create(cls, pred, a, b):
    if isinstance(pred, int):
      pred = IntegerAttr.get(IntegerType.get_signless(64), pred)
    return cls(pred, a, b)


@_ods_cext.register_operation(_Dialect, replace=True)
class ConstantOp(ConstantOp):

  @classmethod
  def create(cls, data_type, value):
    return cls(IntegerAttr.get(data_type, value))
