#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from circt import msft
from typing import Union

DSP = msft.DSP
M20K = msft.M20K


def placement(subpath: Union[str, list[str]],
              devtype: msft.DeviceType,
              x: int,
              y: int,
              num: int = 0):
  loc = msft.PhysLocationAttr.get(devtype, x, y, num)
  if isinstance(subpath, list):
    subpath = "|".join(subpath)
  return (f"loc:{subpath}", loc)
