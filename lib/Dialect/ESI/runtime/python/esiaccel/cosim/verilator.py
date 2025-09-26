#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
from typing import List, Optional, Callable

from .simulator import CosimCollateralDir, Simulator, SourceFiles


class Verilator(Simulator):
  """Run and compile funcs for Verilator."""

  DefaultDriver = CosimCollateralDir / "driver.cpp"

  def __init__(self,
               sources: SourceFiles,
               run_dir: Path,
               debug: bool,
               run_stdout_callback: Optional[Callable[[str], None]] = None,
               run_stderr_callback: Optional[Callable[[str], None]] = None,
               compile_stdout_callback: Optional[Callable[[str], None]] = None,
               compile_stderr_callback: Optional[Callable[[str], None]] = None,
               make_default_logs: bool = True):
    super().__init__(sources=sources,
                     run_dir=run_dir,
                     debug=debug,
                     run_stdout_callback=run_stdout_callback,
                     run_stderr_callback=run_stderr_callback,
                     compile_stdout_callback=compile_stdout_callback,
                     compile_stderr_callback=compile_stderr_callback,
                     make_default_logs=make_default_logs)
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
        "--exe",
        "--build",
        "-j",
        "0",
        "--output-split",
        "--autoflush",
        "--assert",
        str(Verilator.DefaultDriver),
    ]
    cflags = [
        "-DTOP_MODULE=" + self.sources.top,
    ]
    if self.debug:
      cmd += [
          "--trace-fst", "--trace-params", "--trace-structs",
          "--trace-underscore"
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
