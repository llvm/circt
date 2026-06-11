#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A tiny indentation-aware writer for emitting C++ source.

Factored out of the codegen so the type/module emitters don't have to thread
an `indent` string or hand-write trailing newlines and doubled `{{` / `}}`
braces.
"""

from contextlib import contextmanager
from typing import List


class _CppWriter:
  """Accumulate C++ source with automatic indentation.

  Indentation is tracked as a stack of two-space levels so the emit code
  never has to thread an `indent` string or hand-write trailing newlines and
  doubled `{{` / `}}` braces. The three primitives are:

    * `line(text)` -- write one line at the current indent (blank if empty).
    * `block(header)` -- a context manager that writes `header {`, indents its
      body, then closes with `}` (plus an optional `tail`, e.g. `;`).
    * `access(label)` -- write an access specifier (`public:` / `private:`)
      one level out from the members it introduces.

    `write(text)` appends pre-formatted text verbatim (used for the file
    preamble and the irregular window emitters); `indented()` opens a bare
    indent (e.g. a single-statement loop body) without braces. The verbatim
    `write` also lets a `_CppWriter` stand in for a `TextIO` sink.
  """

  def __init__(self, level: int = 0) -> None:
    self._parts: List[str] = []
    self._level = level

  def line(self, text: str = "") -> None:
    self._parts.append(f"{'  ' * self._level}{text}\n" if text else "\n")

  def lines(self, *texts: str) -> None:
    for text in texts:
      self.line(text)

  def write(self, text: str) -> None:
    """Append `text` verbatim (no indentation). Mirrors `TextIO.write` so the
    writer can be passed to emitters that build pre-indented strings."""
    self._parts.append(text)

  def access(self, label: str) -> None:
    # Access specifiers conventionally sit one indent level out from the
    # members they introduce.
    outer = max(self._level - 1, 0)
    self._parts.append(f"{'  ' * outer}{label}\n")

  @contextmanager
  def indented(self):
    self._level += 1
    try:
      yield
    finally:
      self._level -= 1

  @contextmanager
  def block(self, header: str, *, tail: str = ""):
    self.line(f"{header} {{")
    self._level += 1
    try:
      yield
    finally:
      self._level -= 1
      self.line(f"}}{tail}")

  def getvalue(self) -> str:
    return "".join(self._parts)
