#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from pathlib import Path
import logging
import shutil
import subprocess

import pytest

_logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
HW_DIR = ROOT_DIR / "hw"
SW_DIR = ROOT_DIR / "sw"


def require_tool(tool: str) -> None:
  if shutil.which(tool) is None:
    pytest.skip(f"Required tool not found in PATH: {tool}")


def require_env(var_name: str) -> str:
  value = os.environ.get(var_name)
  if not value:
    pytest.skip(f"Required environment variable not set: {var_name}")
  return value


def run_cmd(cmd, **kwargs) -> str:
  """Run a command, capture stdout, and return it. Raises on failure."""
  _logger.info("run_cmd: %s", cmd)
  result = subprocess.run(cmd,
                          check=True,
                          capture_output=True,
                          text=True,
                          **kwargs)
  _logger.debug("stdout: %s", result.stdout)
  _logger.debug("stderr: %s", result.stderr)
  return result.stdout


def check_lines(stdout: str, expected: list[str]) -> None:
  """Assert that every expected substring appears in stdout in order."""
  remaining = stdout
  for line in expected:
    idx = remaining.find(line)
    assert idx >= 0, \
        f"Expected output not found: {line!r}"
    remaining = remaining[idx + len(line):]
