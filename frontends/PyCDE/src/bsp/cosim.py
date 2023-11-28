#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import AppID, Clock, Input
from ..module import Module, generator
from ..system import System
from ..types import types
from .. import esi

from ..circt import ir
from ..circt.dialects import esi as raw_esi

from pathlib import Path

__root_dir__ = Path(__file__).parent.parent


def CosimBSP(user_module):
  """Wrap and return a cosimulation 'board support package' containing
  'user_module'"""

  class top(Module):
    clk = Clock()
    rst = Input(types.int(1))

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)
      raw_esi.ServiceInstanceOp(result=[],
                                appID=AppID("cosim")._appid,
                                service_symbol=None,
                                impl_type=ir.StringAttr.get("cosim"),
                                inputs=[ports.clk.value, ports.rst.value])

      System.current().add_packaging_step(esi.package)

  return top
