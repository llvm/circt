#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from circt import msft
from typing import Union

from pycde.devicedb import PhysLocation, LogicLockedRegion

DSP = msft.DSP
M20K = msft.M20K


def placement(subpath: Union[str, list[str]],
              devtype: msft.PrimitiveType,
              x: int,
              y: int,
              num: int = 0):
  loc = PhysLocation(devtype, x, y, num)
  if isinstance(subpath, list):
    subpath = "|".join(subpath)
  return (f"loc:{subpath}", loc)


def logic_locked_region(subpath: Union[str, list[str]], region_name: str,
                        x_min: int, x_max: int, y_min: int, y_max: int):
  if isinstance(subpath, list):
    subpath = "|".join(subpath)
  region = LogicLockedRegion(region_name, x_min, x_max, y_min, y_max)
  return (f"loc:{subpath}", region)
