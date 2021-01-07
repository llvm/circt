#!/usr/bin/env python3

# ===- circt-rtl-sim.py - CIRCT simulation driver -------------*- python -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//
#
# Script to drive CIRCT simulation tests.
#
# ===-----------------------------------------------------------------------===//

import argparse
import os
import subprocess
import sys

ThisFileDir = os.path.dirname(__file__)


class Questa:
    """Run and compile funcs for Questasim."""

    DefaultDriver = "driver.sv"

    def __init__(self, path, args):
        if os.path.exists(path) and os.path.isfile(path):
            self.path = os.path.dirname(path)
        else:
            self.path = path
        self.args = args

    def compile(self, sources):
        vlog = os.path.join(self.path, "vlog")
        args = [vlog, "-sv"] + sources
        if self.args.gui:
          args.append('+acc')
        return subprocess.run(args)

    def run(self, cycles, simargs):
        if self.args.no_default_driver:
            top = self.args.top
        else:
            top = "driver"

        vsim = os.path.join(self.path, "vsim")
        # Note: vsim exit codes say nothing about the test run's pass/fail even if
        # $fatal is encountered in the simulation.
        if self.args.gui:
          cmd = [vsim, top, "-gui", "-voptargs=\"+acc\""]
        else:
          cmd = [vsim, top, "-batch", "-do", "run -all"]
        if cycles >= 0:
          cmd.append(f"+cycles={cycles}")
        return subprocess.run(cmd + simargs.split())


class Verilator:
    """Run and compile funcs for Verilator."""

    DefaultDriver = "driver.cpp"

    def __init__(self, args):
        if "VERILATOR_PATH" in os.environ:
          self.verilator = os.environ["VERILATOR_PATH"]
        else:
          self.verilator = args.sim
        self.top = args.top
        if args.objdir != "":
          self.ObjDir = args.objdir
        else:
          self.ObjDir = os.path.basename(args.sources[0]) + ".obj_dir"

    def compile(self, sources):
        return subprocess.run([self.verilator, "--cc", "--top-module", self.top,
                               "-sv", "--build", "--exe",
                               "--Mdir", self.ObjDir] + sources)

    def run(self, cycles, args):
        exe = os.path.join(self.ObjDir, "V" + self.top)
        cmd = [exe]
        if cycles >= 0:
          cmd.append(f"--cycles")
          cmd.append(str(cycles))
        cmd += args.split()
        print(f"Running: {cmd}")
        sys.stdout.flush()
        return subprocess.run(cmd)


def __main__(args):
    defaultSim = ""
    if "DEFAULT_SIM" in os.environ:
        defaultSim = os.environ["DEFAULT_SIM"]

    argparser = argparse.ArgumentParser(
        description="RTL simulation runner for CIRCT")

    argparser.add_argument("--sim", type=str, default="verilator",
                           help="Name of the RTL simulator (if in PATH) to use" +
                           " or path to an executable.")
    argparser.add_argument("--no-compile", dest="no_compile", action='store_true',
                           help="Don't compile the simulation.")
    argparser.add_argument("--no-run", dest="no_run", action='store_true',
                           help="Don't run the simulation.")
    argparser.add_argument("--gui", dest="gui", action='store_true',
                           help="Bring up the GUI to run.")
    argparser.add_argument("--top", type=str, default="top",
                           help="Name of top module to run.")
    argparser.add_argument("--objdir", type=str, default="",
                           help="(Verilator) Select an 'obj_dir' to use. Must" +
                           " be different from other tests in the same" +
                           " directory. Defaults to 'sources[0].obj_dir'.")
    argparser.add_argument("--simargs", type=str, default="",
                           help="Simulation arguments string.")
    argparser.add_argument("--no-default-driver", dest="no_default_driver", action='store_true',
                           help="Do not use the standard top module/drivers.")
    argparser.add_argument("--cycles", type=int, default=-1,
                           help="Number of cycles to run the simulator. " +
                                " -1 means don't stop.")

    argparser.add_argument("sources", nargs="+",
                           help="The list of source files to be included.")

    if len(args) <= 1:
        argparser.print_help()
        return
    args = argparser.parse_args(args[1:])

    # Break up simulator string
    simParts = os.path.split(args.sim)
    simName = simParts[1]

    if simName in ["questa", "vsim", "vlog", "vopt"]:
        sim = Questa(simParts[0], args)

    elif simName == "verilator":
        sim = Verilator(args)
    else:
        print(f"Could not determine simulator from '{args.sim}'",
              file=sys.stderr)
        return 1

    if not args.no_default_driver:
      args.sources.append(os.path.join(ThisFileDir, sim.DefaultDriver))

    if not args.no_compile:
        rc = sim.compile(args.sources)
        if rc.returncode != 0:
            return rc
    if not args.no_run:
        rc = sim.run(args.cycles, args.simargs)
        return rc.returncode
    return 0


if __name__ == '__main__':
    sys.exit(__main__(sys.argv))
