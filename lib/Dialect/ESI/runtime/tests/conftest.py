#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

collect_ignore_glob = [
    "integration/hw/*.py",
]


def _looks_like_runtime_root(root: Path) -> bool:
  markers = [
      root / "cmake" / "esiaccelConfig.cmake",
      root / "cpp" / "cmake" / "esiaccelConfig.cmake",
      root / "cpp" / "include" / "esi" / "Accelerator.h",
      root / "lib" / "libESICppRuntime.so",
      root / "lib" / "libESICppRuntime.dylib",
      root / "tests" / "cpp",
      root / "bin",
  ]
  return any(marker.exists() for marker in markers)


def get_runtime_root() -> Path:
  """Determine the root directory of the ESI runtime installation. Since we need
  to support testing in a bunch of different configurations and environments,
  just semi-brute-force the search."""
  import esiaccel

  # Keep the unresolved build-tree path first. The integrated CIRCT build uses
  # symlinks back into the source tree for Python sources, and resolve() would
  # otherwise discard the build location where the runtime libraries live.
  package_file = Path(esiaccel.__file__)
  candidates = [
      package_file.parent.parent,
      package_file.parent.parent.parent,
      package_file.resolve().parent.parent,
      package_file.resolve().parent.parent.parent,
  ]

  seen = set()
  for candidate in candidates:
    if candidate in seen:
      continue
    seen.add(candidate)
    if _looks_like_runtime_root(candidate):
      return candidate

  raise FileNotFoundError("Could not determine ESI runtime root directory")
