#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, Optional, Sequence

import pytest

from esiaccel.cosim.simulator import SourceFiles
from esiaccel.cosim.verilator import Verilator

# xdist note: Verilator builds are CPU/memory heavy. Prefer `-n 2` or `-n 4`
# instead of `-n auto` to avoid oversubscribing the host.

ROOT_DIR = Path(__file__).resolve().parent
HW_DIR = ROOT_DIR / "hw"
SW_DIR = ROOT_DIR / "sw"


@contextlib.contextmanager
def chdir(path: Path):
  old_cwd = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(old_cwd)


@contextlib.contextmanager
def temp_env(overrides: Dict[str, Optional[str]]):
  original: Dict[str, Optional[str]] = {}
  for key, value in overrides.items():
    original[key] = os.environ.get(key)
    if value is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = value
  try:
    yield
  finally:
    for key, value in original.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value


def require_tool(tool: str) -> None:
  if shutil.which(tool) is None:
    pytest.skip(f"Required tool not found in PATH: {tool}")


def resolve_runtime_root() -> Optional[Path]:
  env_root = os.environ.get("ESI_RUNTIME_ROOT")
  if env_root:
    return Path(env_root)

  repo_root = ROOT_DIR
  while repo_root.parent != repo_root:
    if (repo_root / "build").exists() and (repo_root / "lib").exists():
      break
    repo_root = repo_root.parent

  build_roots = [
      repo_root / "build" / "default", repo_root / "build" / "release"
  ]
  lib_names = [
      "libESICppRuntime.so", "libESICppRuntime.dylib", "ESICppRuntime.dll"
  ]
  for build_root in build_roots:
    if not build_root.exists():
      continue
    for lib_name in lib_names:
      if (build_root / "lib" / lib_name).exists():
        return build_root
      if (build_root / "lib64" / lib_name).exists():
        return build_root

  runtime_root = ROOT_DIR.parent.parent
  if (runtime_root / "cpp").exists() and (runtime_root / "python").exists():
    return runtime_root
  return None


def require_runtime_root() -> Path:
  runtime_root = resolve_runtime_root()
  if runtime_root is None:
    pytest.skip("ESI runtime root not found; set ESI_RUNTIME_ROOT")
  return runtime_root


def run_hw_script(script: Path, output_dir: Path, args: Sequence[str]) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  subprocess.run(
      [sys.executable, str(script),
       str(output_dir), *args],
      check=True,
      cwd=output_dir,
  )


def build_simulator(sources_dir: Path, run_dir: Path) -> Verilator:
  sources = SourceFiles("ESI_Cosim_Top")
  hw_dir = sources_dir / "hw"
  sources.add_dir(hw_dir if hw_dir.exists() else sources_dir)
  return Verilator(sources, run_dir, debug=False)


def run_cosim(sim: Verilator, inner_cmd: Sequence[str]) -> None:
  with chdir(sim.run_dir):
    rc = sim.compile()
    assert rc == 0
    ret = sim.run(list(inner_cmd))  # type: ignore[arg-type]
    assert ret == 0


def run_cosim_with_env(sim: Verilator, inner_cmd: Sequence[str],
                       env: Dict[str, Optional[str]]) -> None:
  with temp_env(env):
    run_cosim(sim, inner_cmd)


@pytest.mark.parametrize(
    "name, hw_script, sw_script, hw_args, env",
    [
        (
            "esi_test",
            HW_DIR / "esi_test.py",
            SW_DIR / "esi_test.py",
            [],
            None,
        ),
        (
            "esi_test_manifest_mmio",
            HW_DIR / "esi_test.py",
            SW_DIR / "esi_test.py",
            [],
            {
                "ESI_COSIM_MANIFEST_MMIO": "1"
            },
        ),
        (
            "esi_advanced",
            HW_DIR / "esi_advanced.py",
            SW_DIR / "esi_advanced.py",
            [],
            None,
        ),
        (
            "esi_ram",
            HW_DIR / "esi_ram.py",
            SW_DIR / "esi_ram.py",
            [],
            None,
        ),
        (
            "loopback",
            HW_DIR / "loopback.py",
            SW_DIR / "loopback.py",
            [],
            None,
        ),
    ],
)
def test_cosim_python(name: str, hw_script: Path, sw_script: Path,
                      hw_args: Sequence[str],
                      env: Optional[Dict[str, Optional[str]]],
                      tmp_path: Path) -> None:
  require_tool("verilator")
  hw_out = tmp_path / name
  run_hw_script(hw_script, hw_out, hw_args)

  sim = build_simulator(hw_out, tmp_path / f"{name}_run")
  inner_cmd = [sys.executable, str(sw_script), "cosim", "env"]

  if env:
    run_cosim_with_env(sim, inner_cmd, env)
  else:
    run_cosim(sim, inner_cmd)


@pytest.mark.parametrize(
    "name, bsp, commands",
    [
        (
            "esitester",
            "cosim",
            [
                ["esitester", "-v", "cosim", "env", "callback", "-i", "5"],
                ["esitester", "cosim", "env", "streaming_add"],
                ["esitester", "cosim", "env", "streaming_add", "-t"],
                ["esitester", "cosim", "env", "translate_coords"],
                [
                    "esitester",
                    "cosim",
                    "env",
                    "serial_coords",
                    "-n",
                    "40",
                    "-b",
                    "33",
                ],
                ["esiquery", "cosim", "env", "telemetry"],
            ],
        ),
        (
            "esitester_dma",
            "cosim_dma",
            [
                ["esitester", "cosim", "env", "hostmem"],
                ["esitester", "cosim", "env", "dma", "-w", "-r"],
                ["esiquery", "cosim", "env", "telemetry"],
            ],
        ),
    ],
)
def test_cosim_esitester(name: str, bsp: str, commands: Iterable[Sequence[str]],
                         tmp_path: Path) -> None:
  require_tool("verilator")
  require_tool("esitester")
  require_tool("esiquery")

  hw_out = tmp_path / name
  run_hw_script(HW_DIR / "esitester.py", hw_out, [bsp])

  sim = build_simulator(hw_out, tmp_path / f"{name}_run")
  with chdir(sim.run_dir):
    rc = sim.compile()
    assert rc == 0
    for cmd in commands:
      ret = sim.run(list(cmd))  # type: ignore[arg-type]
      assert ret == 0


@pytest.mark.parametrize(
    "mode",
    [
        "from_manifest",
        "from_accel",
    ],
)
def test_loopback_cpp_codegen(mode: str, tmp_path: Path) -> None:
  require_tool("verilator")
  require_tool("cmake")

  runtime_root = require_runtime_root()
  hw_out = tmp_path / "loopback_cpp"
  run_hw_script(HW_DIR / "loopback.py", hw_out, [])

  sim = build_simulator(hw_out, tmp_path / "loopback_cpp_run")

  include_dir = tmp_path / "include"
  generated_dir = include_dir / "loopback"
  generated_dir.mkdir(parents=True, exist_ok=True)
  if mode == "from_manifest":
    subprocess.run(
        [
            sys.executable,
            "-m",
            "esiaccel.codegen",
            "--file",
            str(hw_out / "esi_system_manifest.json"),
            "--output-dir",
            str(generated_dir),
        ],
        check=True,
    )
  else:
    run_cosim(
        sim,
        [
            sys.executable,
            "-m",
            "esiaccel.codegen",
            "--platform",
            "cosim",
            "--connection",
            "env",
            "--output-dir",
            str(generated_dir),
        ],
    )

  build_dir = tmp_path / f"loopback-build-{mode}"
  subprocess.run(
      [
          "cmake",
          "-S",
          str(SW_DIR),
          "-B",
          str(build_dir),
          f"-DLOOPBACK_GENERATED_DIR={include_dir}",
          f"-DESI_RUNTIME_ROOT={runtime_root}",
      ],
      check=True,
  )
  subprocess.run(
      ["cmake", "--build",
       str(build_dir), "--target", "loopback_test"],
      check=True,
  )

  inner_cmd = [str(build_dir / "loopback_test"), "cosim", "env"]
  run_cosim(sim, inner_cmd)
