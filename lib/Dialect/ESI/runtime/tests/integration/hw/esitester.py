# ===- esitester.py - accelerator for testing ESI functionality -----------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
# Build the esitester accelerator using the included esitester library.
#
# ===----------------------------------------------------------------------===//
import sys

from pycde import System
from pycde.bsp import get_bsp

from esiaccel.esitester import EsiTester

if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2])
  s = System(bsp(EsiTester), name="EsiTester", output_directory=sys.argv[1])
  s.compile()
  s.package()
