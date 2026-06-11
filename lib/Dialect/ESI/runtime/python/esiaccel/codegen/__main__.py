#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Entry point for `python -m esiaccel.codegen`."""

import sys

from . import run

if __name__ == "__main__":
  sys.exit(run())
