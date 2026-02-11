#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil

import pytest

collect_ignore_glob = [
    "integration/hw/*.py",
    "integration/sw/*.py",
]


def require_tool(tool: str) -> None:
  if shutil.which(tool) is None:
    pytest.skip(f"Required tool not found in PATH: {tool}")


def require_env(var_name: str) -> str:
  value = os.environ.get(var_name)
  if not value:
    pytest.skip(f"Required environment variable not set: {var_name}")
  return value
