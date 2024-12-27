#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from .cosim import CosimBSP
from .xrt import XrtBSP


def get_bsp(name: Optional[str] = None):
  if name is None or name == "cosim":
    return CosimBSP
  elif name == "xrt":
    return XrtBSP
  elif name == "xrt_cosim":
    from .xrt import XrtCosimBSP
    return XrtCosimBSP
  else:
    raise ValueError(f"Unknown bsp type: {name}")
