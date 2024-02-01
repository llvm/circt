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
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import textwrap
import time
from typing import List

CosimCollateralDir = Path(os.path.dirname(os.path.realpath(__file__)))


def is_port_open(port) -> bool:
  """Check if a TCP port is open locally."""
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  result = sock.connect_ex(('127.0.0.1', port))
  sock.close()
  return True if result == 0 else False


class SourceFiles:

  def __init__(self, top: str) -> None:
    # User source files.
    self.user: List[Path] = []
    # DPI shared objects.
    self.dpi_so: List[str] = ["EsiCosimDpiServer"]
    # DPI SV files.
    self.dpi_sv: List[Path] = [
        CosimCollateralDir / "Cosim_DpiPkg.sv",
        CosimCollateralDir / "Cosim_Endpoint.sv",
        CosimCollateralDir / "Cosim_Manifest.sv",
        CosimCollateralDir / "Cosim_MMIO.sv",
    ]
    # Name of the top module.
    self.top = top

  def add_dir(self, dir: Path):
    """Add all the RTL files in a directory to the source list."""
    for file in sorted(dir.iterdir()):
      if file.is_file() and (file.suffix == ".sv" or file.suffix == ".v"):
        self.user.append(file)

  def dpi_so_paths(self) -> List[Path]:
    """Return a list of all the DPI shared object files."""

    def find_so(name: str) -> Path:
      for path in os.environ["LD_LIBRARY_PATH"].split(":"):
        if os.name == "nt":
          so = Path(path) / f"{name}.dll"
        else:
          so = Path(path) / f"lib{name}.so"
        if so.exists():
          return so
      raise FileNotFoundError(f"Could not find {name}.so in LD_LIBRARY_PATH")

    return [find_so(name) for name in self.dpi_so]

  @property
  def rtl_sources(self) -> List[Path]:
    """Return a list of all the RTL source files."""
    return self.dpi_sv + self.user


class Simulator:

  # Some RTL simulators don't use stderr for error messages. Everything goes to
  # stdout. Boo! They should feel bad about this. Also, they can specify that
  # broken behavior by overriding this.
  UsesStderr = True

  def __init__(self, sources: SourceFiles, run_dir: Path, debug: bool):
    self.sources = sources
    self.run_dir = run_dir
    self.debug = debug

  def compile_command(self) -> List[str]:
    """Compile the sources. Returns the exit code of the simulation compiler."""
    assert False, "Must be implemented by subclass"

  def compile(self) -> int:
    cp = subprocess.run(self.compile_command(), capture_output=True, text=True)
    self.run_dir.mkdir(parents=True, exist_ok=True)
    open(self.run_dir / "compile_stdout.log", "w").write(cp.stdout)
    open(self.run_dir / "compile_stderr.log", "w").write(cp.stderr)
    if cp.returncode != 0:
      print("====== Compilation failure:")
      if self.UsesStderr:
        print(cp.stderr)
      else:
        print(cp.stdout)
    return cp.returncode

  def run_command(self) -> List[str]:
    """Return the command to run the simulation."""
    assert False, "Must be implemented by subclass"

  def run(self, inner_command: str) -> int:
    """Start the simulation then run the command specified. Kill the simulation
    when the command exits."""

    # 'simProc' is accessed in the finally block. Declare it here to avoid
    # syntax errors in that block.
    simProc = None
    try:
      # Open log files
      self.run_dir.mkdir(parents=True, exist_ok=True)
      simStdout = open(self.run_dir / "sim_stdout.log", "w")
      simStderr = open(self.run_dir / "sim_stderr.log", "w")

      # Erase the config file if it exists. We don't want to read
      # an old config.
      portFileName = self.run_dir / "cosim.cfg"
      if os.path.exists(portFileName):
        os.remove(portFileName)

      # Run the simulation.
      simEnv = os.environ.copy()
      if self.debug:
        simEnv["COSIM_DEBUG_FILE"] = "cosim_debug.log"
      simProc = subprocess.Popen(self.run_command(),
                                 stdout=simStdout,
                                 stderr=simStderr,
                                 env=simEnv,
                                 cwd=self.run_dir,
                                 preexec_fn=os.setsid)
      simStderr.close()
      simStdout.close()

      # Get the port which the simulation RPC selected.
      checkCount = 0
      while (not os.path.exists(portFileName)) and \
              simProc.poll() is None:
        time.sleep(0.1)
        checkCount += 1
        if checkCount > 200:
          raise Exception(f"Cosim never wrote cfg file: {portFileName}")
      port = -1
      while port < 0:
        portFile = open(portFileName, "r")
        for line in portFile.readlines():
          m = re.match("port: (\\d+)", line)
          if m is not None:
            port = int(m.group(1))
        portFile.close()

      # Wait for the simulation to start accepting RPC connections.
      checkCount = 0
      while not is_port_open(port):
        checkCount += 1
        if checkCount > 200:
          raise Exception(f"Cosim RPC port ({port}) never opened")
        if simProc.poll() is not None:
          raise Exception("Simulation exited early")
        time.sleep(0.05)

      # Run the inner command, passing the connection info via environment vars.
      testEnv = os.environ.copy()
      testEnv["ESI_COSIM_PORT"] = str(port)
      testEnv["ESI_COSIM_HOST"] = "localhost"
      return subprocess.run(inner_command, cwd=os.getcwd(),
                            env=testEnv).returncode
    finally:
      # Make sure to stop the simulation no matter what.
      if simProc:
        os.killpg(os.getpgid(simProc.pid), signal.SIGINT)
        # Allow the simulation time to flush its outputs.
        try:
          simProc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
          # If the simulation doesn't exit of its own free will, kill it.
          simProc.kill()


class Verilator(Simulator):
  """Run and compile funcs for Verilator."""

  DefaultDriver = CosimCollateralDir / "driver.cpp"

  def __init__(self, sources: SourceFiles, run_dir: Path, debug: bool):
    super().__init__(sources, run_dir, debug)

    self.verilator = "verilator"
    if "VERILATOR_PATH" in os.environ:
      self.verilator = os.environ["VERILATOR_PATH"]

  def compile_command(self) -> List[str]:
    cmd: List[str] = [
        self.verilator,
        "--cc",
        "--top-module",
        self.sources.top,
        "-DSIMULATION",
        "-sv",
        "--build",
        "--exe",
        "--assert",
        str(Verilator.DefaultDriver),
    ]
    cflags = []
    if self.debug:
      cmd += ["--trace", "--trace-params", "--trace-structs"]
      cflags.append("-DTRACE")
    if len(cflags) > 0:
      cmd += ["-CFLAGS", " ".join(cflags)]
    if len(self.sources.dpi_so) > 0:
      cmd += ["-LDFLAGS", " ".join(["-l" + so for so in self.sources.dpi_so])]
    cmd += [str(p) for p in self.sources.rtl_sources]
    return cmd

  def run_command(self):
    exe = Path.cwd() / "obj_dir" / ("V" + self.sources.top)
    return [str(exe)]


class Questa(Simulator):
  """Run and compile funcs for Questasim."""

  DefaultDriver = CosimCollateralDir / "driver.sv"

  # Questa doesn't use stderr for error messages. Everything goes to stdout.
  UsesStderr = False

  def compile_command(self) -> List[str]:
    cmd = [
        "vlog",
        "-sv",
        "+define+TOP_MODULE=" + self.sources.top,
        "+define+SIMULATION",
        str(Questa.DefaultDriver),
    ]
    cmd += [str(p) for p in self.sources.rtl_sources]
    return cmd

  def run_command(self) -> List[str]:
    vsim = "vsim"
    # Note: vsim exit codes say nothing about the test run's pass/fail even
    # if $fatal is encountered in the simulation.
    cmd = [
        vsim,
        "driver",
        "-batch",
        "-do",
        "run -all",
    ]
    for lib in self.sources.dpi_so_paths():
      svLib = os.path.splitext(lib)[0]
      cmd.append("-sv_lib")
      cmd.append(svLib)
    if len(self.sources.dpi_so) > 0:
      cmd.append("-cpppath")
      cmd.append("/usr/bin/clang++")
    return cmd

  def run(self, inner_command: str) -> int:
    """Override the Simulator.run() to add a soft link in the run directory (to
    the work directory) before running vsim the usual way."""

    # Create a soft link to the work directory.
    workDir = self.run_dir / "work"
    if not workDir.exists():
      os.symlink(Path(os.getcwd()) / "work", workDir)

    # Run the simulation.
    return super().run(inner_command)


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
  argparser.add_argument("--debug",
                         action="store_true",
                         help="Enable debug output.")
  argparser.add_argument("--source",
                         help="Directories containing the source files.",
                         nargs="+",
                         default=["hw"])

  argparser.add_argument("inner_cmd",
                         nargs=argparse.REMAINDER,
                         help="Command to run in the simulation environment.")

  if len(args) <= 1:
    argparser.print_help()
    return
  args = argparser.parse_args(args[1:])

  sources = SourceFiles(args.top)
  for src in args.source:
    sources.add_dir(Path(src))

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

  rc = sim.compile()
  if rc != 0:
    return rc
  return sim.run(args.inner_cmd[1:])


if __name__ == '__main__':
  sys.exit(__main__(sys.argv))
