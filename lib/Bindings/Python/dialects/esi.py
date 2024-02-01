#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from . import esi
from .. import ir
from .. import support
from .._mlir_libs._circt._esi import *
from ..dialects._ods_common import _cext as _ods_cext
from ._esi_ops_gen import *
from ._esi_ops_gen import _Dialect
from typing import Dict, List, Optional, Sequence, Type


class ChannelSignaling:
  ValidReady = 0
  FIFO0 = 1


@_ods_cext.register_operation(_Dialect, replace=True)
class RequestConnectionOp(RequestConnectionOp):

  @property
  def clientNamePath(self) -> List[str]:
    return [
        ir.StringAttr(x).value
        for x in ir.ArrayAttr(self.attributes["clientNamePath"])
    ]


@_ods_cext.register_operation(_Dialect, replace=True)
class RandomAccessMemoryDeclOp(RandomAccessMemoryDeclOp):

  @property
  def innerType(self):
    return ir.TypeAttr(self.attributes["innerType"])


@_ods_cext.register_operation(_Dialect, replace=True)
class ESIPureModuleOp(ESIPureModuleOp):

  def add_entry_block(self):
    if len(self.body.blocks) > 0:
      raise IndexError('The module already has an entry block')
    self.body.blocks.append()
    return self.body.blocks[0]
