#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from enum import Enum


class Mode(Enum):
  User = 0
  Supervisor = 1
  Hypervisor = 2
  Machine = 3
