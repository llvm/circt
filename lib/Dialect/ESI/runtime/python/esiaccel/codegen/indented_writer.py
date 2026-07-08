#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A tiny indentation-aware writer for emitting source text.

Lets the type/module emitters avoid threading an `indent` string or
hand-writing trailing newlines and doubled `{{` / `}}` braces. Text is
streamed straight to the underlying output sink as it is produced --
nothing is buffered.
"""

from contextlib import contextmanager
from typing import Iterator, TextIO


class IndentedWriter:
  """Stream indented source text to an output sink.

  Wraps a `TextIO`-like `out` and tracks indentation as a stack of two-space
  levels, so emit code never has to thread an `indent` string or hand-write
  trailing newlines and doubled `{{` / `}}` braces. Every call writes through
  to `out` immediately. The primitives are:

    * `line(text)` -- write one line at the current indent (blank if empty).
    * `block(header)` -- a context manager that writes `header {`, indents its
      body, then closes with `}` (plus an optional `tail`, e.g. `;`).
    * `access(label)` -- write an access specifier (`public:` / `private:`)
      one level out from the members it introduces.

  `write(text)` streams `text` verbatim (no indentation); it mirrors
  `TextIO.write` so an `IndentedWriter` can itself stand in for a sink.
  `indented()` opens a bare indent (e.g. a single-statement loop body)
  without braces.
  """

  def __init__(self, out: TextIO, level: int = 0) -> None:
    self._out = out
    self._level = level

  def line(self, text: str = "") -> None:
    self._out.write(f"{'  ' * self._level}{text}\n" if text else "\n")

  def lines(self, *texts: str) -> None:
    for text in texts:
      self.line(text)

  def write(self, text: str) -> None:
    """Write `text` verbatim (no indentation). Mirrors `TextIO.write` so the
    writer can be passed to emitters that build pre-indented strings."""
    self._out.write(text)

  def access(self, label: str) -> None:
    # Access specifiers conventionally sit one indent level out from the
    # members they introduce.
    outer = max(self._level - 1, 0)
    self._out.write(f"{'  ' * outer}{label}\n")

  @contextmanager
  def indented(self) -> Iterator[None]:
    self._level += 1
    try:
      yield
    finally:
      self._level -= 1

  @contextmanager
  def block(self, header: str, *, tail: str = "") -> Iterator[None]:
    self.line(f"{header} {{")
    self._level += 1
    try:
      yield
    finally:
      self._level -= 1
      self.line(f"}}{tail}")
