#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Code generation from ESI manifests to source code.

This package only re-exports the public API. The implementation lives in the
sibling modules:

  * `generator` -- the `Generator` base, `CppGenerator`, the C++ type planner /
    emitter (`CppTypePlanner` / `CppTypeEmitter`), and the `run` CLI entry point.
  * `indented_writer` -- the `IndentedWriter` used by the emitters.
  * `ports` -- the per-port-kind strategy table and its rendering helpers.
"""

from .generator import (Generator, CppGenerator, CppTypePlanner, CppTypeEmitter,
                        run)

__all__ = [
    "Generator",
    "CppGenerator",
    "CppTypePlanner",
    "CppTypeEmitter",
    "run",
]
