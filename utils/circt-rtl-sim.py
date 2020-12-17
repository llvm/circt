#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


class Questa:
  """Run and compile funcs for Questasim."""

  def __init__(self, path):
    if os.path.exists(path) and os.path.isfile(path):
      self.path = os.path.dirname(path)
    else:
      self.path = path

  def compile(self, sources):
    vlog = os.path.join(self.path, "vlog")
    return subprocess.run([vlog, "-sv"] + sources)

  def run(self, top, args):
    vsim = os.path.join(self.path, "vsim")
    # Note: vsim exit codes say nothing about the test run's pass/fail even if
    # $fatal is encountered in the simulation.
    return subprocess.run(
      [vsim, top, "-batch", "-do", "run -all"] + args.split())


class Verilator:
  """Run and compile funcs for Verilator."""

  def __init__(self, path, top):
    self.verilator = path
    self.top = top

  def compile(self, sources):
    return subprocess.run([self.verilator, "--cc", "--top-module", self.top,
                           "-sv", "--build", "--exe"] + sources)

  def run(self, top, args):
    exe = os.path.join("obj_dir", "V" + top)
    return subprocess.run([exe] + args.split())


def __main__(args):
  defaultSim = ""
  if "DEFAULT_SIM" in os.environ:
    defaultSim = os.environ["DEFAULT_SIM"]

  argparser = argparse.ArgumentParser(
      description="RTL simulation runner for CIRCT")

  argparser.add_argument("--sim", type=str, default=defaultSim,
                         help="Name of the RTL simulator (if in PATH) to use" +
                         " or path to an executable.")
  argparser.add_argument("--no-compile", type=bool,
                         help="Don't compile the simulation.")
  argparser.add_argument("--no-run", type=bool,
                         help="Don't run the simulation.")
  argparser.add_argument("--top", type=str, default="top",
                         help="Name of top module to run")
  argparser.add_argument("--simargs", type=str, default="",
                         help="Simulation arguments string")

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
    sim = Questa(simParts[0])
  elif simName == "verilator":
    sim = Verilator(args.sim, args.top)
  else:
    print(f"Could not determine simulator from '{args.sim}'",
          file=sys.stderr)
    return 1
  if not args.no_compile:
    rc = sim.compile(args.sources)
    if rc.returncode != 0:
      return rc
  if not args.no_run:
    rc = sim.run(args.top, args.simargs)
    return rc.returncode
  return 0


if __name__ == '__main__':
  sys.exit(__main__(sys.argv))
