#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import comb
from ..dialects._ods_common import _cext as _ods_cext
from ..ir import IntegerAttr, IntegerType, OpView
from ..support import NamedValueOpView, get_value
from ._comb_ops_gen import *
from ._comb_ops_gen import _Dialect


# Sugar classes for the various possible verions of ICmpOp.
class ICmpOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]

  def __init__(self, predicate, data_type, input_port_mapping={}, **kwargs):
    predicate = IntegerAttr.get(IntegerType.get_signless(64), predicate)
    super().__init__(ICmpOp, data_type, input_port_mapping, [predicate],
                     **kwargs)


def CompareOp(predicate):

  def decorated(cls):

    class _Class(cls):

      @staticmethod
      def create(lhs=None, rhs=None):
        mapping = {}
        if lhs:
          mapping["lhs"] = lhs
        if rhs:
          mapping["rhs"] = rhs
        if len(mapping) == 0:
          result_type = IntegerType.get_signless(1)
        else:
          result_type = None
        return ICmpOpBuilder(predicate, result_type, mapping)

    return _Class

  return decorated


@CompareOp(0)
class EqOp(OpView):
  pass


@CompareOp(1)
class NeOp(OpView):
  pass


@CompareOp(2)
class LtSOp(OpView):
  pass


@CompareOp(3)
class LeSOp(OpView):
  pass


@CompareOp(4)
class GtSOp(OpView):
  pass


@CompareOp(5)
class GeSOp(OpView):
  pass


@CompareOp(6)
class LtUOp(OpView):
  pass


@CompareOp(7)
class LeUOp(OpView):
  pass


@CompareOp(8)
class GtUOp(OpView):
  pass


@CompareOp(9)
class GeUOp(OpView):
  pass


# Builder base classes for non-variadic unary and binary ops.
class UnaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["input"]

  def result_names(self):
    return ["result"]


def UnaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, input=None, result_type=None):
      mapping = {"input": input} if input else {}
      return UnaryOpBuilder(cls, result_type, mapping)

  return _Class


class ExtractOpBuilder(UnaryOpBuilder):

  def __init__(self, low_bit, data_type, input_port_mapping={}, **kwargs):
    low_bit = IntegerAttr.get(IntegerType.get_signless(32), low_bit)
    super().__init__(comb.ExtractOp, data_type, input_port_mapping, [],
                     [low_bit], **kwargs)


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


def BinaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, lhs=None, rhs=None, result_type=None):
      mapping = {}
      if lhs:
        mapping["lhs"] = lhs
      if rhs:
        mapping["rhs"] = rhs
      return BinaryOpBuilder(cls, result_type, mapping)

  return _Class


# Base classes for the variadic ops.
def VariadicOp(base):

  class _Class(base):

    @classmethod
    def create(cls, *args):
      return cls([get_value(a) for a in args])

  return _Class


# Base class for miscellaneous ops that can't be abstracted but should provide a
# create method for uniformity.
def CreatableOp(base):

  class _Class(base):

    @classmethod
    def create(cls, *args, **kwargs):
      return cls(*args, **kwargs)

  return _Class


# Sugar classes for the various non-variadic unary ops.
@_ods_cext.register_operation(_Dialect, replace=True)
class ExtractOp(ExtractOp):

  @staticmethod
  def create(low_bit, result_type, input=None):
    mapping = {"input": input} if input else {}
    return ExtractOpBuilder(low_bit,
                            result_type,
                            mapping,
                            needs_result_type=True)


@_ods_cext.register_operation(_Dialect, replace=True)
@UnaryOp
class ParityOp(ParityOp):
  pass


# Sugar classes for the various non-variadic binary ops.
@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class DivSOp(DivSOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class DivUOp(DivUOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class ModSOp(ModSOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class ModUOp(ModUOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class ShlOp(ShlOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class ShrSOp(ShrSOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class ShrUOp(ShrUOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@BinaryOp
class SubOp(SubOp):
  pass


# Sugar classes for the variadic ops.
@_ods_cext.register_operation(_Dialect, replace=True)
@VariadicOp
class AddOp(AddOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@VariadicOp
class MulOp(MulOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@VariadicOp
class AndOp(AndOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@VariadicOp
class OrOp(OrOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@VariadicOp
class XorOp(XorOp):
  pass


@_ods_cext.register_operation(_Dialect, replace=True)
@VariadicOp
class ConcatOp(ConcatOp):
  pass


# Sugar classes for miscellaneous ops.
@_ods_cext.register_operation(_Dialect, replace=True)
@CreatableOp
class MuxOp(MuxOp):
  pass
