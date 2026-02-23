#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import functools
import inspect
import logging
import multiprocessing
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
from esiaccel.accelerator import Accelerator

from .simulator import get_simulator, Simulator, SourceFiles

LogMatcher = Union[str, Pattern[str], Callable[[str, str], bool]]

_logger = logging.getLogger("esiaccel.cosim.pytest")
_DEFAULT_FAILURE_EXCLUDE = re.compile(r"^\s*\d+\s+errors?\b", re.IGNORECASE)
_DEFAULT_FAILURE_PATTERN = re.compile(r"\berror\b", re.IGNORECASE)


@dataclass(frozen=True)
class CosimPytestConfig:
  pycde_script: Union[str, Path]
  args: Sequence[str] = ("{tmp_dir}",)
  simulator: str = "verilator"
  top: str = "ESI_Cosim_Top"
  debug: bool = False
  timeout_s: Optional[float] = None
  failure_matchers: Sequence[LogMatcher] = field(
      default_factory=lambda: [_DEFAULT_FAILURE_PATTERN])
  warning_matchers: Sequence[LogMatcher] = field(default_factory=list)
  failure_exclude: Sequence[Union[str, Pattern[str]]] = field(
      default_factory=lambda: [_DEFAULT_FAILURE_EXCLUDE])


@dataclass
class _ClassCompileCache:
  sources_dir: Path
  obj_dir: Path


@dataclass
class _ChildResult:
  success: bool
  traceback: str = ""
  failure_lines: Sequence[str] = field(default_factory=list)
  warning_lines: Sequence[str] = field(default_factory=list)


@contextlib.contextmanager
def _chdir(path: Path):
  old_cwd = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(old_cwd)


def _as_patterns(
    items: Sequence[Union[str, Pattern[str]]]) -> Sequence[Pattern[str]]:
  patterns = []
  for item in items:
    if isinstance(item, str):
      patterns.append(re.compile(item, re.IGNORECASE))
    else:
      patterns.append(item)
  return patterns


def _line_matches(matchers: Sequence[LogMatcher], line: str,
                  stream: str) -> bool:
  for matcher in matchers:
    if isinstance(matcher, str):
      if re.search(matcher, line, re.IGNORECASE):
        return True
    elif isinstance(matcher, re.Pattern):
      if matcher.search(line):
        return True
    elif matcher(line, stream):
      return True
  return False


def _scan_logs(
    stdout_lines: Sequence[str],
    stderr_lines: Sequence[str],
    config: CosimPytestConfig,
) -> tuple[list[str], list[str]]:
  failure_excludes = _as_patterns(config.failure_exclude)
  failures: list[str] = []
  warnings: list[str] = []

  for stream, lines in (("stdout", stdout_lines), ("stderr", stderr_lines)):
    for line in lines:
      if any(p.search(line) for p in failure_excludes):
        continue
      tagged = f"[{stream}] {line}"
      if _line_matches(config.failure_matchers, line, stream):
        failures.append(tagged)
      if config.warning_matchers and _line_matches(config.warning_matchers,
                                                   line, stream):
        warnings.append(tagged)

  return failures, warnings


def _render_args(args: Sequence[str], tmp_dir: Path) -> list[str]:
  return [arg.format(tmp_dir=tmp_dir) for arg in args]


def _create_simulator(config: CosimPytestConfig, sources_dir: Path,
                      run_dir: Path) -> Simulator:
  sources = SourceFiles(config.top)
  hw_dir = sources_dir / "hw"
  sources.add_dir(hw_dir if hw_dir.exists() else sources_dir)

  return get_simulator(config.simulator, sources, run_dir, config.debug)


def _run_hw_script(config: CosimPytestConfig, tmp_dir: Path) -> Path:
  script = Path(config.pycde_script).resolve()
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
          [sys.executable, "-m", "esiaccel.codegen",
           "--file", str(manifest_path),
           "--output-dir", str(generated_dir)],
          check=True,
          cwd=tmp_dir,
      )
    except subprocess.CalledProcessError:
      # Codegen is optional for tests that don't use C++ artifacts
      pass
  return tmp_dir


def _resolve_injected_params(
    target: Callable[..., Any],
    kwargs: Dict[str, Any],
    host: str,
    port: int,
    sources_dir: Optional[Path] = None,
) -> Dict[str, Any]:
  sig = inspect.signature(target)
  updated = dict(kwargs)
  accepts_host = "host" in sig.parameters or "hostname" in sig.parameters
  accepts_port = "port" in sig.parameters
  if accepts_host and "host" in sig.parameters and "host" not in updated:
    updated["host"] = host
  if accepts_host and "hostname" in sig.parameters and "hostname" not in updated:
    updated["hostname"] = host
  if accepts_port and "port" not in updated:
    updated["port"] = port
  if "sources_dir" in sig.parameters and "sources_dir" not in updated and sources_dir is not None:
    updated["sources_dir"] = sources_dir

  for name, param in sig.parameters.items():
    annotation = param.annotation
    is_accel = annotation is Accelerator or name == "accelerator"
    if is_accel and name not in updated:
      conn = esiaccel.connect("cosim", f"{host}:{port}")
      updated[name] = conn.build_accelerator()

  return updated


def _pytest_visible_signature(target: Callable[..., Any]) -> inspect.Signature:
  sig = inspect.signature(target)
  kept_params = []
  for param in sig.parameters.values():
    if param.name in {"host", "hostname", "port", "accelerator", "sources_dir"}:
      continue
    if param.annotation is Accelerator:
      continue
    kept_params.append(param)
  return sig.replace(parameters=kept_params)


def _copy_compiled_obj_dir(compiled_obj_dir: Optional[Path], run_dir: Path):
  if compiled_obj_dir is None:
    return
  dst = run_dir / "obj_dir"
  if dst.exists():
    shutil.rmtree(dst)
  shutil.copytree(compiled_obj_dir, dst)


def _compile_once_for_class(config: CosimPytestConfig) -> _ClassCompileCache:
  compile_root = Path(tempfile.mkdtemp(prefix="esi-pytest-class-compile-"))
  sources_dir = _run_hw_script(config, compile_root)
  compile_dir = compile_root / "compile"
  sim = _create_simulator(config, sources_dir, compile_dir)
  rc = sim.compile()
  if rc != 0:
    raise RuntimeError(f"Simulator compile failed with exit code {rc}")
  return _ClassCompileCache(sources_dir=sources_dir,
                            obj_dir=compile_dir / "obj_dir")


def _run_child(
    result_queue: multiprocessing.Queue,
    target: Callable[..., Any],
    config: CosimPytestConfig,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    class_cache: Optional[_ClassCompileCache],
):
  stdout_lines: list[str] = []
  stderr_lines: list[str] = []

  def on_stdout(line: str):
    stdout_lines.append(line)

  def on_stderr(line: str):
    stderr_lines.append(line)

  sim_proc = None
  sim = None
  try:
    run_root = Path(tempfile.mkdtemp(prefix="esi-pytest-run-"))
    if class_cache is None:
      sources_dir = _run_hw_script(config, run_root)
      run_dir = run_root / f"run-{os.getpid()}"
      sim = _create_simulator(config, sources_dir, run_dir)
      sim._run_stdout_cb = on_stdout
      sim._run_stderr_cb = on_stderr
      sim._compile_stdout_cb = on_stdout
      sim._compile_stderr_cb = on_stderr
      rc = sim.compile()
      if rc != 0:
        raise RuntimeError(f"Simulator compile failed with exit code {rc}")
    else:
      run_dir = run_root / f"run-{os.getpid()}"
      sim = _create_simulator(config, class_cache.sources_dir, run_dir)
      sim._run_stdout_cb = on_stdout
      sim._run_stderr_cb = on_stderr
      _copy_compiled_obj_dir(class_cache.obj_dir, run_dir)

    sim_proc = sim.run_proc()
    injected_kwargs = _resolve_injected_params(target, kwargs, "localhost",
                                               sim_proc.port, sources_dir=sources_dir)
    target(*args, **injected_kwargs)

    failure_lines, warning_lines = _scan_logs(stdout_lines, stderr_lines,
                                              config)
    if failure_lines:
      raise AssertionError("Detected simulator failures:\n" +
                           "\n".join(failure_lines))

    result_queue.put(
        _ChildResult(success=True,
                     warning_lines=warning_lines,
                     failure_lines=failure_lines))
  except Exception:
    result_queue.put(
        _ChildResult(success=False,
                     traceback=traceback.format_exc(),
                     warning_lines=_scan_logs(stdout_lines, stderr_lines,
                                              config)[1]))
  finally:
    if sim_proc is not None and sim_proc.proc.poll() is None:
      sim_proc.force_stop()


def _run_isolated(
    target: Callable[..., Any],
    config: CosimPytestConfig,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    class_cache: Optional[_ClassCompileCache] = None,
):
  ctx = multiprocessing.get_context("fork")
  queue: multiprocessing.Queue = ctx.Queue()
  process = ctx.Process(
      target=_run_child,
      args=(queue, target, config, args, kwargs, class_cache),
  )
  process.start()
  process.join(timeout=config.timeout_s)

  if process.is_alive():
    process.terminate()
    process.join(timeout=1)
    raise AssertionError(
        f"Cosim test timed out after {config.timeout_s} seconds")

  if queue.empty():
    raise RuntimeError(
        f"Cosim child exited without returning a result (exit code: {process.exitcode})"
    )

  result: _ChildResult = queue.get()
  for warning in result.warning_lines:
    _logger.warning("cosim warning: %s", warning)

  if not result.success:
    raise AssertionError(result.traceback)


def _decorate_function(
    target: Callable[..., Any],
    config: CosimPytestConfig,
    class_cache_getter: Optional[Callable[[], _ClassCompileCache]] = None,
) -> Callable[..., Any]:

  @functools.wraps(target)
  def _wrapper(*args, **kwargs):
    cache = class_cache_getter() if class_cache_getter is not None else None
    _run_isolated(target, config, args, kwargs, class_cache=cache)

  setattr(_wrapper, "__signature__", _pytest_visible_signature(target))
  return _wrapper


def _decorate_class(target_cls: type, config: CosimPytestConfig) -> type:
  lock = threading.Lock()
  cache_holder: dict[str, _ClassCompileCache] = {}

  def _get_cache() -> _ClassCompileCache:
    with lock:
      if "cache" not in cache_holder:
        cache_holder["cache"] = _compile_once_for_class(config)
    return cache_holder["cache"]

  for name, member in list(vars(target_cls).items()):
    if name.startswith("test") and callable(member):
      setattr(
          target_cls,
          name,
          _decorate_function(member, config, class_cache_getter=_get_cache),
      )
  return target_cls


def cosim_test(
    pycde_script: Union[str, Path],
    args: Sequence[str] = ("{tmp_dir}",),
    simulator: str = "verilator",
    top: str = "ESI_Cosim_Top",
    debug: bool = False,
    timeout_s: Optional[float] = None,
    failure_matchers: Optional[Sequence[LogMatcher]] = None,
    warning_matchers: Optional[Sequence[LogMatcher]] = None,
    failure_exclude: Optional[Sequence[Union[str, Pattern[str]]]] = None,
):
  config = CosimPytestConfig(
      pycde_script=pycde_script,
      args=args,
      simulator=simulator,
      top=top,
      debug=debug,
      timeout_s=timeout_s,
      failure_matchers=failure_matchers
      if failure_matchers is not None else [_DEFAULT_FAILURE_PATTERN],
      warning_matchers=warning_matchers if warning_matchers is not None else [],
      failure_exclude=failure_exclude
      if failure_exclude is not None else [_DEFAULT_FAILURE_EXCLUDE],
  )

  def _decorator(target):
    if inspect.isclass(target):
      return _decorate_class(target, config)
    if callable(target):
      return _decorate_function(target, config)
    raise TypeError("@cosim_test can decorate functions or classes")

  return _decorator
