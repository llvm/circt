#!/usr/bin/env python3

# ===- esi-cosim.py - ESI cosimulation launch utility --------*- python -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
# Utility script to start a simulation and launch a command to interact with it
# via ESI cosimulation.
#
# ===----------------------------------------------------------------------===//

import argparse
from pathlib import Path
import sys
import textwrap
from typing import Dict, List

from ..cosim.questa import Questa
from ..cosim.verilator import Verilator
from ..cosim.simulator import SourceFiles


def __main__(args):
  argparser = argparse.ArgumentParser(
      description="Wrap a 'inner_cmd' in an ESI cosimulation environment.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent("""
        Notes:
          - For Verilator, libEsiCosimDpiServer.so must be in the dynamic
          library runtime search path (LD_LIBRARY_PATH) and link time path
          (LIBRARY_PATH). If it is installed to a standard location (e.g.
          /usr/lib), this should be handled automatically.
          - This script needs to sit in the same directory as the ESI support
          SystemVerilog (e.g. Cosim_DpiPkg.sv, Cosim_MMIO.sv, etc.). It can,
          however, be soft linked to a different location.
          - The simulator executable(s) must be in your PATH.
      """))

  argparser.add_argument(
      "--sim",
      type=str,
      default="verilator",
      help="Name of the RTL simulator to use or path to an executable.")
  argparser.add_argument("--rundir",
                         default="run",
                         help="Directory in which simulation should be run.")
  argparser.add_argument(
      "--top",
      default="ESI_Cosim_Top",
      help="Name of the 'top' module to use in the simulation.")
  argparser.add_argument("--no-compile",
                         action="store_true",
                         help="Do not run the compile.")
  argparser.add_argument("--debug",
                         action="store_true",
                         help="Enable debug output.")
  argparser.add_argument("--gui",
                         action="store_true",
                         help="Run the simulator in GUI mode (if supported).")
  argparser.add_argument("--source",
                         help="Directories containing the source files.",
                         default="hw")

  argparser.add_argument("inner_cmd",
                         nargs=argparse.REMAINDER,
                         help="Command to run in the simulation environment.")

  argparser.add_argument(
      "--server-only",
      action="store_true",
      help="Only run the cosim server, and do not run any inner command.")

  if len(args) <= 1:
    argparser.print_help()
    return
  args = argparser.parse_args(args[1:])

  sources = SourceFiles(args.top)
  sources.add_dir(Path(args.source))

  if args.sim == "verilator":
    sim = Verilator(sources, Path(args.rundir), args.debug)
  elif args.sim == "questa":
    sim = Questa(sources, Path(args.rundir), args.debug)
  else:
    print("Unknown simulator: " + args.sim)
    print("Supported simulators: ")
    print("  - verilator")
    print("  - questa")
    return 1

  if not args.no_compile:
    rc = sim.compile()
    if rc != 0:
      return rc
  return sim.run(args.inner_cmd[1:], gui=args.gui, server_only=args.server_only)


if __name__ == '__main__':
  sys.exit(__main__(sys.argv))
