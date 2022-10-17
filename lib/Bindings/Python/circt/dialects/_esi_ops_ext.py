from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Type

from circt.dialects import esi
import circt.support as support

import mlir.ir as ir


class RequestToServerConnectionOp:

  @property
  def clientNamePath(self) -> List[str]:
    return [
        ir.StringAttr(x).value
        for x in ir.ArrayAttr(self.attributes["clientNamePath"])
    ]


class RequestToClientConnectionOp:

  @property
  def clientNamePath(self) -> List[str]:
    return [
        ir.StringAttr(x).value
        for x in ir.ArrayAttr(self.attributes["clientNamePath"])
    ]
