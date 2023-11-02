#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import hw
from .._mlir_libs._circt._seq import *
from ..dialects._ods_common import _cext as _ods_cext
from ..ir import IntegerType, OpView, StringAttr
from ..support import BackedgeBuilder, NamedValueOpView
from ._seq_ops_gen import *
from ._seq_ops_gen import _Dialect
from .seq import CompRegOp


# Create a computational register whose input is the given value, and is clocked
# by the given clock. If a reset is provided, the register will be reset by that
# signal. If a reset value is provided, the register will reset to that,
# otherwise it will reset to zero. If name is provided, the register will be
# named.
def reg(value, clock, reset=None, reset_value=None, name=None, sym_name=None):
  from . import hw
  from ..ir import IntegerAttr
  value_type = value.type
  if reset:
    if not reset_value:
      zero = IntegerAttr.get(value_type, 0)
      reset_value = hw.ConstantOp(zero).result
    return CompRegOp.create(value_type,
                            input=value,
                            clk=clock,
                            reset=reset,
                            reset_value=reset_value,
                            name=name,
                            sym_name=sym_name).data.value
  else:
    return CompRegOp.create(value_type,
                            input=value,
                            clk=clock,
                            name=name,
                            sym_name=sym_name).data.value


class CompRegLikeBuilder(NamedValueOpView):

  def result_names(self):
    return ["data"]

  def create_initial_value(self, index, data_type, arg_name):
    if arg_name == "input":
      operand_type = data_type
    else:
      operand_type = IntegerType.get_signless(1)
    return BackedgeBuilder.create(operand_type, arg_name, self)


class CompRegLike:

  def __init__(self,
               data_type,
               input,
               clk,
               clockEnable=None,
               *,
               reset=None,
               reset_value=None,
               power_on_value=None,
               name=None,
               sym_name=None,
               loc=None,
               ip=None):
    operands = [input, clk]
    results = []
    attributes = {}
    results.append(data_type)
    operand_segment_sizes = [1, 1]
    if isinstance(self, CompRegOp):
      if clockEnable is not None:
        raise Exception("Clock enable not supported on compreg")
    elif isinstance(self, CompRegClockEnabledOp):
      if clockEnable is None:
        raise Exception("Clock enable required on compreg.ce")
      operands.append(clockEnable)
      operand_segment_sizes.append(1)
    else:
      assert False, "Class not recognized"
    if reset is not None and reset_value is not None:
      operands.append(reset)
      operands.append(reset_value)
      operand_segment_sizes += [1, 1]
    else:
      operand_segment_sizes += [0, 0]
      operands += [None, None]

    if power_on_value is not None:
      operands.append(power_on_value)
      operand_segment_sizes.append(1)
    else:
      operands.append(None)
      operand_segment_sizes.append(0)
    if name is None:
      attributes["name"] = StringAttr.get("")
    else:
      attributes["name"] = StringAttr.get(name)
    if sym_name is not None:
      attributes["inner_sym"] = hw.InnerSymAttr.get(StringAttr.get(sym_name))

    self._ODS_OPERAND_SEGMENTS = operand_segment_sizes

    OpView.__init__(
        self,
        self.build_generic(
            attributes=attributes,
            results=results,
            operands=operands,
            loc=loc,
            ip=ip,
        ),
    )


class CompRegBuilder(CompRegLikeBuilder):

  def operand_names(self):
    return ["input", "clk"]


@_ods_cext.register_operation(_Dialect, replace=True)
class CompRegOp(CompRegLike, CompRegOp):

  @classmethod
  def create(cls,
             result_type,
             reset=None,
             reset_value=None,
             name=None,
             sym_name=None,
             **kwargs):
    return CompRegBuilder(cls,
                          result_type,
                          kwargs,
                          reset=reset,
                          reset_value=reset_value,
                          name=name,
                          sym_name=sym_name,
                          needs_result_type=True)


class CompRegClockEnabledBuilder(CompRegLikeBuilder):

  def operand_names(self):
    return ["input", "clk", "clockEnable"]


@_ods_cext.register_operation(_Dialect, replace=True)
class CompRegClockEnabledOp(CompRegLike, CompRegClockEnabledOp):

  @classmethod
  def create(cls,
             result_type,
             reset=None,
             reset_value=None,
             name=None,
             sym_name=None,
             **kwargs):
    return CompRegClockEnabledBuilder(cls,
                                      result_type,
                                      kwargs,
                                      reset=reset,
                                      reset_value=reset_value,
                                      name=name,
                                      sym_name=sym_name,
                                      needs_result_type=True)
