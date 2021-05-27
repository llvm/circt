#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generated tablegen dialects end up in the mlir.dialects package for now.
from mlir.dialects._comb_ops_gen import *

from circt.support import NamedValueOpView

from mlir.ir import IntegerAttr, IntegerType


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


class EqOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(0, *args, **kwargs)


class EqOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(0, *args, **kwargs)


class NeOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(1, *args, **kwargs)


class LtSOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(2, *args, **kwargs)


class LeSOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(3, *args, **kwargs)


class GtSOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(4, *args, **kwargs)


class GeSOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(5, *args, **kwargs)


class LtUOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(6, *args, **kwargs)


class LeUOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(7, *args, **kwargs)


class GtUOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(8, *args, **kwargs)


class GeUOp:

  @staticmethod
  def create(*args, **kwargs):
    return ICmpOpBuilder(9, *args, **kwargs)
