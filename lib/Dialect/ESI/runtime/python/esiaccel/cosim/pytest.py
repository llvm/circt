#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pytest integration for ESI cosimulation tests.

Provides the ``@cosim_test`` decorator which automates the full lifecycle of a
cosimulation test: running a PyCDE hardware script, compiling the design with
a simulator (e.g. Verilator), launching the simulator, injecting connection
parameters into the test function, and tearing everything down afterwards.

Decorated functions run in an isolated child process (via ``fork``) so that
simulator state never leaks between tests.  When applied to a class, the
hardware compilation is performed once and shared across all ``test_*`` methods.

Typical usage::

    from esiaccel.cosim.pytest import cosim_test

    @cosim_test("path/to/hw_script.py")
    def test_my_design(conn: AcceleratorConnection):
        # conn is already connected to the running simulator
        ...
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
import functools
import inspect
import logging
import multiprocessing
import multiprocessing.connection
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import Any, Callable, Dict, Optional, Pattern, Sequence, Union

import esiaccel
from esiaccel.accelerator import Accelerator, AcceleratorConnection

from .simulator import get_simulator, Simulator, SourceFiles

LogMatcher = Union[str, Pattern[str], Callable[[str, str], bool]]
SourceGeneratorFunc = Callable[["CosimPytestConfig", Path], Path]
SourceGeneratorArg = Union[str, Path, SourceGeneratorFunc]

_logger = logging.getLogger("esiaccel.cosim.pytest")
_DEFAULT_FAILURE_PATTERN = re.compile(r"\berror\b", re.IGNORECASE)
_DEFAULT_WARN_PATTERN = re.compile(r"\bwarn(ing)?\b", re.IGNORECASE)
# Default per-test wall-clock timeout in seconds.  Matches the 120 s limit
# used by the lit integration test suite (CIRCT_INTEGRATION_TIMEOUT).
_DEFAULT_TIMEOUT_S: float = 120.0


def _get_env_bool(var_name: str, default: bool) -> bool:
  """Read a boolean environment variable.

  Args:
    var_name: Name of the environment variable to read.
    default: Default value if the variable is not set.

  Returns:
    The boolean value of the environment variable, or the default value.
  """
  value = os.environ.get(var_name)
  if value is None:
    return default
  return value.lower() in ("true", "1", "yes", "on")


def _get_env_path(var_name: str) -> Optional[Path]:
  """Read a path environment variable.

  Args:
    var_name: Name of the environment variable to read.

  Returns:
    The path value of the environment variable, or None if not set.
  """
  value = os.environ.get(var_name)
  return Path(value) if value else None


def _get_pytest_run_id() -> str:
  """Get a unique identifier for this pytest invocation.

  For normal runs, uses the current process PID (the pytest process itself).
  For xdist workers, uses the parent PID (the main pytest controller) so that
  all workers share the same top-level run directory.
  """
  if os.environ.get("PYTEST_XDIST_WORKER"):
    # Worker process — use parent (the main pytest controller) for grouping.
    return f"pytest-{os.getppid()}"
  return f"pytest-{os.getpid()}"


def _get_xdist_worker_id() -> Optional[str]:
  """Get the xdist worker ID if running under pytest-xdist.

  Returns the worker ID (e.g., "gw0", "gw1") or None if not using xdist.
  """
  return os.environ.get("PYTEST_XDIST_WORKER")


def _get_test_dir_name(config: CosimPytestConfig) -> str:
  """Build a test directory path including pytest run ID and xdist worker (if present).

  For class-based tests with xdist: "pytest-{pid}/gw{n}/ClassName/test_method".
  For class-based tests without xdist: "pytest-{pid}/ClassName/test_method".
  For function tests with xdist: "pytest-{pid}/gw{n}/test_function".
  For function tests without xdist: "pytest-{pid}/test_function".
  """
  parts = []

  if config.pytest_run_id:
    parts.append(config.pytest_run_id)

  if config.xdist_worker_id:
    parts.append(config.xdist_worker_id)

  if config.class_name:
    parts.append(config.class_name)

  if config.test_name:
    parts.append(config.test_name)

  if not parts:
    return "unknown-test"

  return "/".join(parts)


@dataclass(frozen=True)
class CosimPytestConfig:
  """Immutable configuration for a single cosim test or test class.

  Attributes:
    source_generator: Path to the PyCDE hardware generation script, or a
      callable ``(config, tmp_dir) -> sources_dir``.
    args: Arguments passed to the script; ``{tmp_dir}`` is interpolated.
    simulator: Simulator backend name (e.g. ``"verilator"``).
    top: Top-level module name for the simulator.
    debug: If True, enable verbose simulator output.
    timeout_s: Maximum wall-clock seconds before the test is killed.
    failure_matcher: Pattern applied to simulator output to detect errors.
    warning_matcher: Pattern applied to simulator output to detect warnings.
    tmp_dir_root: Root directory for temporary directories. If None, uses system temp.
    delete_tmp_dir: If True, delete temporary directories after test completion.
    class_name: Name of the test class (for class-based tests). None for function tests.
    test_name: Name of the test function or method.
    pytest_run_id: Unique identifier for the pytest run (e.g., "pytest-12345").
    xdist_worker_id: ID of the xdist worker if running under pytest-xdist (e.g., "gw0").
    save_waveform: If True, dump waveform file. Format depends on backend. Requires debug=True.
  """

  source_generator: SourceGeneratorArg
  args: Sequence[str] = ("{tmp_dir}",)
  simulator: str = "verilator"
  top: str = "ESI_Cosim_Top"
  debug: bool = False
  timeout_s: float = _DEFAULT_TIMEOUT_S
  failure_matcher: Optional[LogMatcher] = _DEFAULT_FAILURE_PATTERN
  warning_matcher: Optional[LogMatcher] = _DEFAULT_WARN_PATTERN
  tmp_dir_root: Optional[Path] = None
  delete_tmp_dir: bool = True
  class_name: Optional[str] = None
  test_name: Optional[str] = None
  pytest_run_id: Optional[str] = None
  xdist_worker_id: Optional[str] = None
  save_waveform: bool = False


@dataclass
class _ClassCompileCache:
  """Cached compilation artifacts shared across methods of a test class."""

  sources_dir: Path
  compile_dir: Path


@dataclass
class _ChildResult:
  """Outcome of a child-process test execution, passed back via queue."""

  success: bool
  traceback: str = ""
  failure_lines: Sequence[str] = field(default_factory=list)
  warning_lines: Sequence[str] = field(default_factory=list)
  stdout_lines: Sequence[str] = field(default_factory=list)
  stderr_lines: Sequence[str] = field(default_factory=list)


@contextlib.contextmanager
def _chdir(path: Path):
  """Context manager that temporarily changes the working directory."""
  old_cwd = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(old_cwd)


def _line_matches(matcher: LogMatcher, line: str, stream: str) -> bool:
  """Return True if *line* matches the given matcher.

  The matcher may be a plain string (regex search), a compiled regex,
  or a callable ``(line, stream) -> bool``.
  """
  if isinstance(matcher, str):
    return bool(re.search(matcher, line, re.IGNORECASE))
  elif isinstance(matcher, re.Pattern):
    return bool(matcher.search(line))
  else:
    return matcher(line, stream)


def _scan_logs(
    stdout_lines: Sequence[str],
    stderr_lines: Sequence[str],
    config: CosimPytestConfig,
) -> tuple[list[str], list[str]]:
  """Scan simulator output for failures and warnings.

  Returns:
    A ``(failures, warnings)`` tuple of tagged log lines.
  """
  failures: list[str] = []
  warnings: list[str] = []

  for stream, lines in (("stdout", stdout_lines), ("stderr", stderr_lines)):
    for line in lines:
      tagged = f"[{stream}] {line}"
      if config.failure_matcher and _line_matches(config.failure_matcher, line,
                                                  stream):
        failures.append(tagged)
      if config.warning_matcher and _line_matches(config.warning_matcher, line,
                                                  stream):
        warnings.append(tagged)

  return failures, warnings


def _render_args(args: Sequence[str], tmp_dir: Path) -> list[str]:
  """Interpolate ``{tmp_dir}`` placeholders in script arguments."""
  return [arg.format(tmp_dir=tmp_dir) for arg in args]


def _generate_sources(config: CosimPytestConfig, tmp_dir: Path) -> Path:
  """Generate hardware sources via a normalized source-generator callable.

  If ``config.source_generator`` is a string/path, it is wrapped into a callable
  that runs the PyCDE script. If it is already callable, it is used directly.
  """
  source_spec = config.source_generator
  if isinstance(source_spec, (str, Path)):
    source_generator: SourceGeneratorFunc = (
        lambda inner_config, inner_tmp_dir: _run_hw_script(
            source_spec, inner_config, inner_tmp_dir))
  else:
    source_generator = source_spec
  return source_generator(config, tmp_dir)


def _create_simulator(config: CosimPytestConfig, sources_dir: Path,
                      run_dir: Path) -> Simulator:
  """Instantiate a ``Simulator`` from the generated source files."""
  sources = SourceFiles(config.top)
  hw_dir = sources_dir / "hw"
  sources.add_dir(hw_dir if hw_dir.exists() else sources_dir)

  return get_simulator(config.simulator, sources, run_dir, config.debug,
                       config.save_waveform)


def _run_hw_script(script_path: Union[str, Path], config: CosimPytestConfig,
                   tmp_dir: Path) -> Path:
  """Execute the PyCDE hardware script and run codegen if a manifest exists.

  Returns:
    The directory containing the generated sources (same as *tmp_dir*).
  """
  script = Path(script_path).resolve()
  script_args = _render_args(config.args, tmp_dir)
  with _chdir(tmp_dir):
    subprocess.run([sys.executable, str(script), *script_args],
                   check=True,
                   cwd=tmp_dir)

  # Run codegen automatically to generate C++ artifacts from manifest, if present.
  manifest_path = tmp_dir / "esi_system_manifest.json"
  if manifest_path.exists():
    generated_dir = tmp_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    try:
      subprocess.run(
          [
              sys.executable, "-m", "esiaccel.codegen", "--file",
              str(manifest_path), "--output-dir",
              str(generated_dir)
          ],
          check=True,
          cwd=tmp_dir,
      )
    except subprocess.CalledProcessError as e:
      # Codegen is optional for tests that don't use C++ artifacts
      _logger.warning("codegen failed (non-fatal): %s", e)
  return tmp_dir


# Names and annotations that the decorator injects automatically.
_INJECTED_NAMES = {
    "host", "hostname", "port", "sources_dir", "conn", "accelerator"
}
_INJECTED_ANNOTATIONS = frozenset({Accelerator, AcceleratorConnection})


def _is_injected_param(name: str, annotation: Any) -> bool:
  """Return True if *name*/*annotation* will be supplied by the decorator."""
  return name in _INJECTED_NAMES or annotation in _INJECTED_ANNOTATIONS


def _resolve_injected_params(
    target: Callable[..., Any],
    kwargs: Dict[str, Any],
    host: str,
    port: int,
    sources_dir: Optional[Path] = None,
) -> Dict[str, Any]:
  """Build the keyword arguments to inject into the test function.

  Inspects the target's signature and automatically supplies ``host``,
  ``port``, ``sources_dir``, ``AcceleratorConnection``, or ``Accelerator``
  parameters that the test declares but the caller did not provide.
  """
  sig = inspect.signature(target)
  updated = dict(kwargs)

  for name, param in sig.parameters.items():
    if name in updated:
      continue
    if name in ("host", "hostname"):
      updated[name] = host
    elif name == "port":
      updated[name] = port
    elif name == "sources_dir" and sources_dir is not None:
      updated[name] = sources_dir
    elif param.annotation is AcceleratorConnection or name == "conn":
      updated[name] = esiaccel.connect("cosim", f"{host}:{port}")
    elif param.annotation is Accelerator or name == "accelerator":
      conn = esiaccel.connect("cosim", f"{host}:{port}")
      updated[name] = conn.build_accelerator()

  return updated


def _visible_signature(target: Callable[..., Any]) -> inspect.Signature:
  """Return a signature with injected parameters removed.

  Pytest uses function signatures to determine fixture requirements.  This
  hides the parameters that the decorator injects (``host``, ``port``,
  ``conn``, etc.) so pytest does not try to resolve them as fixtures.
  Uses :func:`_is_injected_param` as the single source of truth.
  """
  sig = inspect.signature(target)
  kept = [
      p for p in sig.parameters.values()
      if not _is_injected_param(p.name, p.annotation)
  ]
  return sig.replace(parameters=kept)


def _copy_compiled_artifacts(compile_dir: Optional[Path], run_dir: Path):
  """Copy pre-compiled simulator artifacts into the per-test run directory.

  Copies the *entire* compile directory so that all backends (Verilator,
  Questa, etc.) find their artefacts regardless of internal layout.
  """
  if compile_dir is None:
    return
  run_dir.mkdir(parents=True, exist_ok=True)
  for item in compile_dir.iterdir():
    dst = run_dir / item.name
    if item.is_dir():
      shutil.copytree(item, dst, dirs_exist_ok=True)
    else:
      shutil.copy2(item, dst)


def _compile_once_for_class(config: CosimPytestConfig) -> _ClassCompileCache:
  """Run the hw script and compile the simulator once for a whole test class.

  The resulting ``_ClassCompileCache`` is reused by each test method to avoid
  redundant compilations.
  """
  # When using a custom tmp dir, create directories directly under it for easier debugging.
  # When using the system temp dir, use mkdtemp for automatic isolation.
  if config.tmp_dir_root is not None:
    # Organize using hierarchical naming: pytest-{pid}/[gw{n}/]ClassName/__compile__
    test_dir = _get_test_dir_name(config)
    compile_root = config.tmp_dir_root / test_dir / "__compile__"
    compile_root.mkdir(parents=True, exist_ok=True)
  else:
    compile_root = Path(tempfile.mkdtemp(prefix="esi-pytest-class-compile-",))
  try:
    sources_dir = _generate_sources(config, compile_root)
    compile_dir = compile_root / "compile"
    sim = _create_simulator(config, sources_dir, compile_dir)
    with _chdir(compile_dir):
      rc = sim.compile()
    if rc != 0:
      raise RuntimeError(f"Simulator compile failed with exit code {rc}")
    return _ClassCompileCache(sources_dir=sources_dir, compile_dir=compile_dir)
  except Exception:
    if config.delete_tmp_dir and not config.debug:
      shutil.rmtree(compile_root, ignore_errors=True)
    raise


def _run_child(
    result_pipe: multiprocessing.connection.Connection,
    target: Callable[..., Any],
    config: CosimPytestConfig,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    class_cache: Optional[_ClassCompileCache],
):
  """Entry point for the forked child process.

  Compiles (or reuses cached compilation), starts the simulator, injects
  connection parameters, calls the test function, scans logs for failures,
  and sends a ``_ChildResult`` through *result_pipe*.
  """
  # Keep compile and run output separate.  Only run-time output is scanned
  # by the failure/warning matchers; compile failures are caught via exit code.
  compile_stdout: list[str] = []
  compile_stderr: list[str] = []
  stdout_lines: list[str] = []
  stderr_lines: list[str] = []

  def on_stdout(line: str):
    stdout_lines.append(line)

  def on_stderr(line: str):
    stderr_lines.append(line)

  sim_proc = None
  sim = None
  run_root = None
  run_dir = None
  try:
    # When using a custom tmp dir, create directories directly under it for easier debugging.
    # When using the system temp dir, use mkdtemp for automatic isolation.
    if config.tmp_dir_root is not None:
      # Organize under test class and function names for easy identification
      test_dir = _get_test_dir_name(config)
      run_root = config.tmp_dir_root / test_dir
      run_root.mkdir(parents=True, exist_ok=True)
    else:
      run_root = Path(tempfile.mkdtemp(prefix="esi-pytest-run-",))
    if class_cache is None:
      sources_dir = _generate_sources(config, run_root)
      run_dir = run_root
      sim = _create_simulator(config, sources_dir, run_dir)
      sim._run_stdout_cb = on_stdout
      sim._run_stderr_cb = on_stderr
      sim._compile_stdout_cb = compile_stdout.append
      sim._compile_stderr_cb = compile_stderr.append
      os.chdir(run_dir)
      rc = sim.compile()
      if rc != 0:
        raise RuntimeError(f"Simulator compile failed with exit code {rc}")
    else:
      sources_dir = class_cache.sources_dir
      run_dir = run_root
      sim = _create_simulator(config, sources_dir, run_dir)
      sim._run_stdout_cb = on_stdout
      sim._run_stderr_cb = on_stderr
      _copy_compiled_artifacts(class_cache.compile_dir, run_dir)

    os.chdir(run_dir)
    sim_proc = sim.run_proc()
    injected_kwargs = _resolve_injected_params(target,
                                               kwargs,
                                               "localhost",
                                               sim_proc.port,
                                               sources_dir=sources_dir)
    target(*args, **injected_kwargs)

    failure_lines, warning_lines = _scan_logs(stdout_lines, stderr_lines,
                                              config)
    if failure_lines:
      raise AssertionError("Detected simulator failures:\n" +
                           "\n".join(failure_lines))

    # Combine compile + run output for the diagnostic dump, but only
    # runtime output was fed to the failure/warning matchers above.
    all_stdout = compile_stdout + stdout_lines
    all_stderr = compile_stderr + stderr_lines
    result_pipe.send(
        _ChildResult(success=True,
                     warning_lines=warning_lines,
                     failure_lines=failure_lines,
                     stdout_lines=all_stdout,
                     stderr_lines=all_stderr))
  except Exception:
    all_stdout = compile_stdout + stdout_lines
    all_stderr = compile_stderr + stderr_lines
    result_pipe.send(
        _ChildResult(success=False,
                     traceback=traceback.format_exc(),
                     warning_lines=_scan_logs(stdout_lines, stderr_lines,
                                              config)[1],
                     stdout_lines=all_stdout,
                     stderr_lines=all_stderr))
  finally:
    result_pipe.close()
    if sim_proc is not None and sim_proc.proc.poll() is None:
      sim_proc.force_stop()
    # When a custom tmp_dir_root is set, keep directories by default for debugging.
    # When using system temp, delete by default to avoid clutter.
    should_delete = config.delete_tmp_dir and not config.debug
    if run_root is not None and should_delete:
      shutil.rmtree(run_root, ignore_errors=True)
    # Force-exit the forked child.  Non-daemon threads spawned by gRPC (or
    # other native libraries) can prevent a normal exit even after all Python
    # work has finished.  The result is already in the pipe, so this is safe.
    os._exit(0)


def _run_isolated(
    target: Callable[..., Any],
    config: CosimPytestConfig,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    class_cache: Optional[_ClassCompileCache] = None,
):
  """Fork a child process to run *target* and wait for its result.

  Handles timeouts, collects warnings, and re-raises any failure from
  the child as an ``AssertionError`` in the parent.
  """
  try:
    ctx = multiprocessing.get_context("fork")
  except ValueError:
    import pytest as _pytest
    _pytest.skip("fork start method unavailable on this platform")
  reader, writer = ctx.Pipe(duplex=False)
  process = ctx.Process(
      target=_run_child,
      args=(writer, target, config, args, kwargs, class_cache),
  )
  process.start()
  writer.close()  # Parent only reads.

  # Wait for the result with an optional timeout.  We poll the pipe first
  # so that we can detect a child crash even before join() returns.
  result: Optional[_ChildResult] = None
  if reader.poll(timeout=config.timeout_s):
    result = reader.recv()
  reader.close()
  process.join(timeout=10)
  if process.is_alive():
    process.terminate()
    process.join(timeout=5)

  if result is None:
    if config.timeout_s is not None:
      raise AssertionError(
          f"Cosim test timed out after {config.timeout_s} seconds")
    raise RuntimeError(
        f"Cosim child exited without returning a result (exit code: {process.exitcode})"
    )

  # Always surface simulation logs for post-mortem debugging.
  for line in result.stdout_lines:
    _logger.debug("sim stdout: %s", line)
  for line in result.stderr_lines:
    _logger.debug("sim stderr: %s", line)
  for warning in result.warning_lines:
    _logger.warning("cosim warning: %s", warning)

  if not result.success:
    parts = [result.traceback]
    if result.stdout_lines:
      parts.append("\n=== Simulator stdout ===")
      parts.extend(result.stdout_lines[-200:])
    if result.stderr_lines:
      parts.append("\n=== Simulator stderr ===")
      parts.extend(result.stderr_lines[-200:])
    raise AssertionError("\n".join(parts))


def _decorate_function(
    target: Callable[..., Any],
    config: CosimPytestConfig,
    class_cache_getter: Optional[Callable[[], _ClassCompileCache]] = None,
) -> Callable[..., Any]:
  """Wrap a single test function so it runs inside ``_run_isolated``."""
  # Set the test name in the config
  test_config = replace(config, test_name=target.__name__)

  @functools.wraps(target)
  def _wrapper(*args, **kwargs):
    cache = class_cache_getter() if class_cache_getter is not None else None
    _run_isolated(target, test_config, args, kwargs, class_cache=cache)

  setattr(_wrapper, "__signature__", _visible_signature(target))
  return _wrapper


def _decorate_class(target_cls: type, config: CosimPytestConfig) -> type:
  """Wrap every ``test_*`` method of a class with cosim isolation.

  Compilation is performed once (lazily, on first method invocation) and
  the resulting artifacts are shared across all methods via a thread-safe
  cache.
  """
  # Set the class name in the config for all methods
  class_config = replace(config, class_name=target_cls.__name__)
  lock = threading.Lock()
  cache_holder: dict[str, _ClassCompileCache] = {}

  def _get_cache() -> _ClassCompileCache:
    with lock:
      if "cache" not in cache_holder:
        cache_holder["cache"] = _compile_once_for_class(class_config)
    return cache_holder["cache"]

  for name, member in list(vars(target_cls).items()):
    if name.startswith("test") and callable(member):
      setattr(
          target_cls,
          name,
          _decorate_function(member,
                             class_config,
                             class_cache_getter=_get_cache),
      )
  return target_cls


def cosim_test(
    source_generator: SourceGeneratorArg,
    args: Sequence[str] = ("{tmp_dir}",),
    simulator: str = "verilator",
    top: str = "ESI_Cosim_Top",
    debug: Optional[bool] = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    failure_matcher: Optional[LogMatcher] = _DEFAULT_FAILURE_PATTERN,
    warning_matcher: Optional[LogMatcher] = _DEFAULT_WARN_PATTERN,
    tmp_dir_root: Optional[Path] = None,
    delete_tmp_dir: Optional[bool] = None,
    save_waveform: Optional[bool] = None,
):
  """Decorator that turns a function or class into a cosimulation test.

  The decorated target is executed in a forked child process with a freshly
  compiled and running simulator.  Connection parameters (``host``, ``port``,
  ``acc``, etc.) are injected automatically based on the function signature.

  When applied to a class, the hardware script is run and compiled once;
  each ``test_*`` method gets its own simulator process but skips
  recompilation.

  Args:
    source_generator: Path to the PyCDE script that generates the hardware, or
      a callable ``(config, tmp_dir) -> sources_dir`` that generates
      sources directly.
    args: Arguments forwarded to the script; ``{tmp_dir}`` is interpolated
        with the temporary build directory.
    simulator: Simulator backend (default ``"verilator"``).
    top: Top-level module name.
    debug: Enable verbose simulator output. Defaults to the value of the
      ``ESIACCEL_PYTEST_DEBUG`` environment variable if set, otherwise False.
    timeout_s: Wall-clock timeout in seconds (default 120).
    failure_matcher: Pattern to detect errors in simulator output.
    warning_matcher: Pattern to detect warnings in simulator output.
    tmp_dir_root: Root directory for temporary test files. Defaults to the value
      of the ``ESIACCEL_PYTEST_TMP_DIR`` environment variable if set, otherwise
      uses the system temporary directory. Run directories are organized in a
      hierarchy to avoid collisions during parallel execution:

      - Normal run: ``pytest-{pid}/ClassName/test_method/`` or
        ``pytest-{pid}/test_function/``
      - xdist parallel: ``pytest-{pid}/gw0/ClassName/test_method/`` (gw0, gw1, etc.
        for different workers)
    delete_tmp_dir: Whether to delete temporary directories after test
      completion. When tmp_dir_root is set (custom debugging directory),
      directories are kept by default; set ``ESIACCEL_PYTEST_DELETE_TMP_DIR=true``
      to delete them. When using the system temp directory, defaults to True
      (delete to avoid clutter). Always False when debug mode is enabled.
    save_waveform: Whether to save waveform dumps (format depends on the
      simulator backend, e.g. FST for Verilator, VCD for Questa). Requires
      debug mode to be enabled. Defaults to the value of the
      ``ESIACCEL_PYTEST_SAVE_WAVEFORM`` environment variable if set, otherwise
      False.
  """
  # Use environment variables as defaults if not explicitly provided
  if debug is None:
    debug = _get_env_bool("ESIACCEL_PYTEST_DEBUG", False)
  if tmp_dir_root is None:
    tmp_dir_root = _get_env_path("ESIACCEL_PYTEST_TMP_DIR")
  if delete_tmp_dir is None:
    # When using a custom tmp_dir_root (provided explicitly or via env),
    # keep directories by default for debugging.  When using the system
    # temp directory, delete them by default.  The env var overrides either.
    default_delete = tmp_dir_root is None
    delete_tmp_dir = _get_env_bool("ESIACCEL_PYTEST_DELETE_TMP_DIR",
                                   default_delete)
  if save_waveform is None:
    save_waveform = _get_env_bool("ESIACCEL_PYTEST_SAVE_WAVEFORM", False)

  # Waveform dumping requires debug mode
  if save_waveform and not debug:
    _logger.warning(
        "save_waveform requires debug mode to be enabled; disabling waveform dumping"
    )
    save_waveform = False

  # Get the pytest run ID for unique test isolation
  pytest_run_id = _get_pytest_run_id()
  xdist_worker_id = _get_xdist_worker_id()

  config = CosimPytestConfig(
      source_generator=source_generator,
      args=args,
      simulator=simulator,
      top=top,
      debug=debug,
      timeout_s=timeout_s,
      failure_matcher=failure_matcher,
      warning_matcher=warning_matcher,
      tmp_dir_root=tmp_dir_root,
      delete_tmp_dir=delete_tmp_dir,
      pytest_run_id=pytest_run_id,
      xdist_worker_id=xdist_worker_id,
      save_waveform=save_waveform,
  )

  def _decorator(target):
    if inspect.isclass(target):
      return _decorate_class(target, config)
    if callable(target):
      return _decorate_function(target, config)
    raise TypeError("@cosim_test can decorate functions or classes")

  return _decorator
