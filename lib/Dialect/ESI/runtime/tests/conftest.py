#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

collect_ignore_glob = [
    "integration/hw/*.py",
]


def get_runtime_root() -> Path:
  import esiaccel
  p = Path(esiaccel.__file__).resolve().parent.parent
  if (p / "lib").exists():
    return p
  p = Path(esiaccel.__file__).resolve().parent.parent.parent
  if p.exists():
    return p
  raise FileNotFoundError("Could not determine ESI runtime root directory")
