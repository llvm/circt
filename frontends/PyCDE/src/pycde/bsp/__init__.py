#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from .cosim import CosimBSP, CosimBSP_DMA
from .xrt import XrtBSP


def get_bsp(name: Optional[str] = None):
  if name is None or name == "cosim":
    return CosimBSP
  elif name == "cosim_dma":
    return CosimBSP_DMA
  elif name == "xrt":
    return XrtBSP
  else:
    raise ValueError(f"Unknown bsp type: {name}")
