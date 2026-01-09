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
from typing import Dict, List, Optional, Callable, IO
import threading

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
        CosimCollateralDir / "Cosim_CycleCount.sv",
        CosimCollateralDir / "Cosim_Manifest.sv",
    ]
    # Name of the top module.
    self.top = top

  def add_file(self, file: Path):
    """Add a single RTL file to the source list."""
    if file.is_file():
      self.user.append(file)
    else:
      raise FileNotFoundError(f"File {file} does not exist")

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

  def __init__(self,
               proc: subprocess.Popen,
               port: int,
               threads: Optional[List[threading.Thread]] = None,
               gui: bool = False):
    self.proc = proc
    self.port = port
    self.threads: List[threading.Thread] = threads or []
    self.gui = gui

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

    # Join reader threads (they should exit once pipes are closed).
    for t in self.threads:
      t.join()


class Simulator:

  # Some RTL simulators don't use stderr for error messages. Everything goes to
  # stdout. Boo! They should feel bad about this. Also, they can specify that
  # broken behavior by overriding this.
  UsesStderr = True

  def __init__(self,
               sources: SourceFiles,
               run_dir: Path,
               debug: bool,
               run_stdout_callback: Optional[Callable[[str], None]] = None,
               run_stderr_callback: Optional[Callable[[str], None]] = None,
               compile_stdout_callback: Optional[Callable[[str], None]] = None,
               compile_stderr_callback: Optional[Callable[[str], None]] = None,
               make_default_logs: bool = True,
               macro_definitions: Optional[Dict[str, str]] = None):
    """Simulator base class.

    Optional sinks can be provided for capturing output. If not provided,
    the simulator will write to log files in `run_dir`.

    Args:
      sources: SourceFiles describing RTL/DPI inputs.
      run_dir: Directory where build/run artifacts are placed.
      debug: Enable cosim debug mode.
      run_stdout_callback: Line-based callback for runtime stdout.
      run_stderr_callback: Line-based callback for runtime stderr.
      compile_stdout_callback: Line-based callback for compile stdout.
      compile_stderr_callback: Line-based callback for compile stderr.
      make_default_logs: If True and corresponding callback is not supplied,
        create log file and emit via internally-created callback.
      macro_definitions: Optional dictionary of macro definitions to be defined
        during compilation.
    """
    self.sources = sources
    self.run_dir = run_dir
    self.debug = debug
    self.macro_definitions = macro_definitions

    # Unified list of any log file handles we opened.
    self._default_files: List[IO[str]] = []

    def _ensure_default(cb: Optional[Callable[[str], None]], filename: str):
      """Return (callback, file_handle_or_None) with optional file creation.

      Behavior:
        * If a callback is provided, return it unchanged with no file.
        * If no callback and make_default_logs is False, return (None, None).
        * If no callback and make_default_logs is True, create a log file and
          return a writer callback plus the opened file handle.
      """
      if cb is not None:
        return cb, None
      if not make_default_logs:
        return None, None
      p = self.run_dir / filename
      p.parent.mkdir(parents=True, exist_ok=True)
      logf = p.open("w+")
      self._default_files.append(logf)

      def _writer(line: str, _lf=logf):
        _lf.write(line + "\n")
        _lf.flush()

      return _writer, logf

    # Initialize all four (compile/run stdout/stderr) uniformly.
    self._compile_stdout_cb, self._compile_stdout_log = _ensure_default(
        compile_stdout_callback, 'compile_stdout.log')
    self._compile_stderr_cb, self._compile_stderr_log = _ensure_default(
        compile_stderr_callback, 'compile_stderr.log')
    self._run_stdout_cb, self._run_stdout_log = _ensure_default(
        run_stdout_callback, 'sim_stdout.log')
    self._run_stderr_cb, self._run_stderr_log = _ensure_default(
        run_stderr_callback, 'sim_stderr.log')

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
    for cmd in cmds:
      ret = self._start_process_with_callbacks(
          cmd,
          env=Simulator.get_env(),
          cwd=None,
          stdout_cb=self._compile_stdout_cb,
          stderr_cb=self._compile_stderr_cb,
          wait=True)
      if isinstance(ret, int) and ret != 0:
        print("====== Compilation failure")

        # If we have the default file loggers, print the compilation logs to
        # console. Else, assume that the user has already captured them.
        if self.UsesStderr:
          if self._compile_stderr_log is not None:
            self._compile_stderr_log.seek(0)
            print(self._compile_stderr_log.read())
        else:
          if self._compile_stdout_log is not None:
            self._compile_stdout_log.seek(0)
            print(self._compile_stdout_log.read())

        return ret
    return 0

  def run_command(self, gui: bool) -> List[str]:
    """Return the command to run the simulation."""
    assert False, "Must be implemented by subclass"

  def run_proc(self, gui: bool = False) -> SimProcess:
    """Run the simulation process. Returns the Popen object and the port which
      the simulation is listening on.

    If user-provided stdout/stderr sinks were supplied in the constructor,
    they are used. Otherwise, log files are created in `run_dir`.
    """
    self.run_dir.mkdir(parents=True, exist_ok=True)

    env_gui = os.environ.get("COSIM_GUI", "0")
    if env_gui != "0":
      gui = True

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
    # Start process with asynchronous output capture.
    proc, threads = self._start_process_with_callbacks(
        rcmd,
        env=simEnv,
        cwd=self.run_dir,
        stdout_cb=self._run_stdout_cb,
        stderr_cb=self._run_stderr_cb,
        wait=False)

    # Get the port which the simulation RPC selected.
    checkCount = 0
    while (not os.path.exists(portFileName)) and \
            proc.poll() is None:
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
      if proc.poll() is not None:
        raise Exception("Simulation exited early")
      time.sleep(0.05)
    return SimProcess(proc=proc, port=port, threads=threads, gui=gui)

  def _start_process_with_callbacks(
      self, cmd: List[str], env: Optional[Dict[str, str]], cwd: Optional[Path],
      stdout_cb: Optional[Callable[[str],
                                   None]], stderr_cb: Optional[Callable[[str],
                                                                        None]],
      wait: bool) -> int | tuple[subprocess.Popen, List[threading.Thread]]:
    """Start a subprocess and stream its stdout/stderr to callbacks.

    If wait is True, blocks until process completes and returns its exit code.
    If wait is False, returns the Popen object (threads keep streaming).
    """
    if os.name == "posix":
      proc = subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              env=env,
                              cwd=cwd,
                              text=True,
                              preexec_fn=os.setsid)
    else:  # windows
      proc = subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              env=env,
                              cwd=cwd,
                              text=True,
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    def _reader(pipe, cb):
      if pipe is None:
        return
      for raw in pipe:
        if raw.endswith('\n'):
          raw = raw[:-1]
        if cb:
          try:
            cb(raw)
          except Exception as e:
            print(f"Exception in simulator output callback: {e}")

    threads: List[threading.Thread] = [
        threading.Thread(target=_reader,
                         args=(proc.stdout, stdout_cb),
                         daemon=True),
        threading.Thread(target=_reader,
                         args=(proc.stderr, stderr_cb),
                         daemon=True),
    ]
    for t in threads:
      t.start()
    if wait:
      for t in threads:
        t.join()
      return proc.wait()
    return proc, threads

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
        ret = subprocess.run(inner_command, cwd=os.getcwd(),
                             env=testEnv).returncode
        if simProc.gui:
          print("GUI mode - waiting for simulator to exit...")
          simProc.proc.wait()
        return ret
    finally:
      if simProc and simProc.proc.poll() is None:
        simProc.force_stop()
