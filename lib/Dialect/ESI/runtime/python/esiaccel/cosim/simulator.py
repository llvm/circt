#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List

_thisdir = Path(__file__).parent
CosimCollateralDir = _thisdir


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
    ]
    # Name of the top module.
    self.top = top

  def add_dir(self, dir: Path):
    """Add all the RTL files in a directory to the source list."""
    for file in sorted(dir.iterdir()):
      if file.is_file() and (file.suffix == ".sv" or file.suffix == ".v"):
        self.user.append(file)
      elif file.is_dir():
        self.add_dir(file)

  def dpi_so_paths(self) -> List[Path]:
    """Return a list of all the DPI shared object files."""

    def find_so(name: str) -> Path:
      for path in Simulator.get_env().get("LD_LIBRARY_PATH", "").split(":"):
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


class SimProcess:

  def __init__(self, proc: subprocess.Popen, port: int):
    self.proc = proc
    self.port = port

  def force_stop(self):
    """Make sure to stop the simulation no matter what."""
    if self.proc:
      os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
      # Allow the simulation time to flush its outputs.
      try:
        self.proc.wait(timeout=1.0)
      except subprocess.TimeoutExpired:
        # If the simulation doesn't exit of its own free will, kill it.
        self.proc.kill()


class Simulator:

  # Some RTL simulators don't use stderr for error messages. Everything goes to
  # stdout. Boo! They should feel bad about this. Also, they can specify that
  # broken behavior by overriding this.
  UsesStderr = True

  def __init__(self, sources: SourceFiles, run_dir: Path, debug: bool):
    self.sources = sources
    self.run_dir = run_dir
    self.debug = debug

  @staticmethod
  def get_env() -> Dict[str, str]:
    """Get the environment variables to locate shared objects."""

    env = os.environ.copy()
    env["LIBRARY_PATH"] = env.get("LIBRARY_PATH", "") + ":" + str(
        _thisdir.parent / "lib")
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + str(
        _thisdir.parent / "lib")
    return env

  def compile_commands(self) -> List[List[str]]:
    """Compile the sources. Returns the exit code of the simulation compiler."""
    assert False, "Must be implemented by subclass"

  def compile(self) -> int:
    cmds = self.compile_commands()
    self.run_dir.mkdir(parents=True, exist_ok=True)
    with (self.run_dir / "compile_stdout.log").open("w") as stdout, (
        self.run_dir / "compile_stderr.log").open("w") as stderr:
      for cmd in cmds:
        stderr.write(" ".join(cmd) + "\n")
        cp = subprocess.run(cmd,
                            env=Simulator.get_env(),
                            capture_output=True,
                            text=True)
        stdout.write(cp.stdout)
        stderr.write(cp.stderr)
        if cp.returncode != 0:
          print("====== Compilation failure:")
          if self.UsesStderr:
            print(cp.stderr)
          else:
            print(cp.stdout)
          return cp.returncode
    return 0

  def run_command(self, gui: bool) -> List[str]:
    """Return the command to run the simulation."""
    assert False, "Must be implemented by subclass"

  def run_proc(self, gui: bool = False) -> SimProcess:
    """Run the simulation process. Returns the Popen object and the port which
      the simulation is listening on."""
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
    simEnv = Simulator.get_env()
    if self.debug:
      simEnv["COSIM_DEBUG_FILE"] = "cosim_debug.log"
      if "DEBUG_PERIOD" not in simEnv:
        # Slow the simulation down to one tick per millisecond.
        simEnv["DEBUG_PERIOD"] = "1"
    rcmd = self.run_command(gui)
    simProc = subprocess.Popen(self.run_command(gui),
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
      if checkCount > 500 and not gui:
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
    return SimProcess(proc=simProc, port=port)

  def run(self,
          inner_command: str,
          gui: bool = False,
          server_only: bool = False) -> int:
    """Start the simulation then run the command specified. Kill the simulation
    when the command exits."""

    # 'simProc' is accessed in the finally block. Declare it here to avoid
    # syntax errors in that block.
    simProc = None
    try:
      simProc = self.run_proc(gui=gui)
      if server_only:
        # wait for user input to kill the server
        input(
            f"Running in server-only mode on port {simProc.port} - Press anything to kill the server..."
        )
        return 0
      else:
        # Run the inner command, passing the connection info via environment vars.
        testEnv = os.environ.copy()
        testEnv["ESI_COSIM_PORT"] = str(simProc.port)
        testEnv["ESI_COSIM_HOST"] = "localhost"
        return subprocess.run(inner_command, cwd=os.getcwd(),
                              env=testEnv).returncode
    finally:
      if simProc:
        simProc.force_stop()
