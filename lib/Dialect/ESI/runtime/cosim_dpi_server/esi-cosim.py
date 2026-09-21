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

from esiaccel.cosim.simulator import (get_simulator, load_macro_definitions,
                                      SourceFiles)


def _parse_macro_definition(value):
  name, separator, macro_value = value.partition("=")
  if not name:
    raise argparse.ArgumentTypeError("macro name cannot be empty")
  return name, macro_value if separator else None


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
  argparser.add_argument(
      "--save-waveform",
      action="store_true",
      help="Save waveform dumps (format depends on simulator). Requires --debug."
  )
  argparser.add_argument("--gui",
                         action="store_true",
                         help="Run the simulator in GUI mode (if supported).")
  argparser.add_argument("--source",
                         help="Directories containing the source files.",
                         default="hw")
  argparser.add_argument(
      "-D",
      "--define",
      dest="macro_definitions",
      action="append",
      default=[],
      metavar="NAME[=VALUE]",
      type=_parse_macro_definition,
      help="Define an RTL macro during compilation. May be specified multiple "
      "times.")
  argparser.add_argument(
      "--define-file",
      dest="macro_definitions_file",
      metavar="FILE",
      help="Read RTL macro definitions from a JSON file: an object mapping "
      "macro name to value, where null defines the macro without one. Useful "
      "when the macros are produced by the same build that generated the "
      "sources. Any -D on the command line overrides what the file defines.")

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

  # Validate that save_waveform requires debug
  if args.save_waveform and not args.debug:
    print("ERROR: --save-waveform requires --debug to be enabled",
          file=sys.stderr)
    return 1

  sources = SourceFiles(args.top)
  sources.add_dir(Path(args.source))

  macro_definitions = {}
  if args.macro_definitions_file is not None:
    macro_definitions.update(
        load_macro_definitions(Path(args.macro_definitions_file)))
  macro_definitions.update(args.macro_definitions)

  sim = get_simulator(args.sim,
                      sources,
                      Path(args.rundir),
                      args.debug,
                      args.save_waveform,
                      macro_definitions=macro_definitions)
  if not args.no_compile:
    rc = sim.compile()
    if rc != 0:
      return rc
  return sim.run(args.inner_cmd[1:], gui=args.gui, server_only=args.server_only)


if __name__ == '__main__':
  sys.exit(__main__(sys.argv))
