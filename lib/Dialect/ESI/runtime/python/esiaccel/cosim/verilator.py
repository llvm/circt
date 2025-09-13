#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
from typing import List

from .simulator import CosimCollateralDir, Simulator, SourceFiles


class Verilator(Simulator):
  """Run and compile funcs for Verilator."""

  DefaultDriver = CosimCollateralDir / "driver.cpp"

  def __init__(self, sources: SourceFiles, run_dir: Path, debug: bool):
    super().__init__(sources, run_dir, debug)

    self.verilator = "verilator"
    if "VERILATOR_PATH" in os.environ:
      self.verilator = os.environ["VERILATOR_PATH"]

  def compile_commands(self) -> List[List[str]]:
    cmd: List[str] = [
        self.verilator,
        "--cc",
        "--top-module",
        self.sources.top,
        "-DSIMULATION",
        "-Wno-TIMESCALEMOD",
        "-Wno-fatal",
        "-sv",
        "--build",
        "--exe",
        "--assert",
        str(Verilator.DefaultDriver),
    ]
    cflags = [
        "-DTOP_MODULE=" + self.sources.top,
    ]
    if self.debug:
      cmd += [
          "--trace", "--trace-params", "--trace-structs", "--trace-underscore"
      ]
      cflags.append("-DTRACE")
    if len(cflags) > 0:
      cmd += ["-CFLAGS", " ".join(cflags)]
    if len(self.sources.dpi_so) > 0:
      cmd += ["-LDFLAGS", " ".join(["-l" + so for so in self.sources.dpi_so])]
    cmd += [str(p) for p in self.sources.rtl_sources]
    return [cmd]

  def run_command(self, gui: bool):
    if gui:
      raise RuntimeError("Verilator does not support GUI mode.")
    exe = Path.cwd() / "obj_dir" / ("V" + self.sources.top)
    return [str(exe)]
