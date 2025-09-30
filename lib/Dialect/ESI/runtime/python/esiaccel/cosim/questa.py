#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
from typing import List

from .simulator import CosimCollateralDir, Simulator


class Questa(Simulator):
  """Run and compile funcs for Questasim."""

  DefaultDriver = CosimCollateralDir / "driver.sv"

  # Questa doesn't use stderr for error messages. Everything goes to stdout.
  UsesStderr = False

  def internal_compile_commands(self) -> List[str]:
    cmds = [
        "onerror { quit -f -code 1 }",
    ]
    sources = self.sources.rtl_sources
    sources.append(Questa.DefaultDriver)
    for src in sources:
      cmds.append(f"vlog -incr +acc -sv +define+TOP_MODULE={self.sources.top}"
                  f" +define+SIMULATION {src.as_posix()}")
    cmds.append(f"vopt -incr driver -o driver_opt +acc")
    return cmds

  def compile_commands(self) -> List[List[str]]:
    with open("compile.do", "w") as f:
      for cmd in self.internal_compile_commands():
        f.write(cmd)
        f.write("\n")
      f.write("quit\n")
    return [
        ["vsim", "-batch", "-do", "compile.do"],
    ]

  def run_command(self, gui: bool) -> List[str]:
    vsim = "vsim"
    # Note: vsim exit codes say nothing about the test run's pass/fail even
    # if $fatal is encountered in the simulation.
    if gui:
      cmd = [
          vsim,
          "driver_opt",
      ]
    else:
      cmd = [
          vsim,
          "driver_opt",
          "-batch",
          "-do",
          "run -all",
      ]
    for lib in self.sources.dpi_so_paths():
      svLib = os.path.splitext(lib)[0]
      cmd.append("-sv_lib")
      cmd.append(svLib)
    return cmd

  def run(self,
          inner_command: str,
          gui: bool = False,
          server_only: bool = False) -> int:
    """Override the Simulator.run() to add a soft link in the run directory (to
    the work directory) before running vsim the usual way."""

    # Create a soft link to the work directory.
    workDir = self.run_dir / "work"
    if not workDir.exists():
      os.symlink(Path(os.getcwd()) / "work", workDir)

    # Run the simulation.
    return super().run(inner_command, gui, server_only=server_only)
