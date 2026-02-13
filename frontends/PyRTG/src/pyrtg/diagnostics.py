#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import ir

import os
import sys

from typing import Optional, List
from dataclasses import dataclass
from enum import Enum


class Verbosity(Enum):
  """
  The verbosity levels for diagnostic output.
  """

  CONCISE = 'concise'
  NORMAL = 'normal'
  VERBOSE = 'verbose'


_pyrtg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_user_code(loc: ir.Location):
  """
  Check if a file is user code (not part of PyRTG library).
  """

  if loc.is_a_name():
    loc = loc.child_loc

  if not loc.is_a_file():
    return True

  # Filter out Python internal frames
  if loc.filename.startswith('<') or loc.filename.startswith('frozen'):
    return False

  # Filter out standard library
  import sys
  if any(
      loc.filename.startswith(p)
      for p in sys.path
      if 'site-packages' not in p and 'dist-packages' not in p):
    return False

  # Filter out PyRTG library frames
  return not os.path.abspath(loc.filename).startswith(_pyrtg_dir)


class Color(Enum):
  """
  Color codes for terminal output.
  """

  NONE = ''
  RED = '\033[91m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  BLUE = '\033[94m'
  MAGENTA = '\033[95m'
  CYAN = '\033[96m'


def colored(text: str, color: Color, bold: bool = False) -> str:
  """
  Return colored text string if stderr supports colors.
  
  :param text: The text to colorize.
  :param color: ANSI color code to apply (use Color class constants).
  :param bold: If True, make the text bold (default: False).
  :return: The text wrapped in ANSI color codes if stderr is a TTY, otherwise
           the original text unchanged.
  """

  if not sys.stderr.isatty():
    return text

  prefix = '\033[1m' if bold else ''
  reset_formatting = '\033[0m'
  return f"{prefix}{color.value}{text}{reset_formatting}"


def colored_severity(severity: ir.DiagnosticSeverity) -> str:
  """
  Return colored severity string if stderr supports colors.
  """

  if severity == ir.DiagnosticSeverity.ERROR:
    return colored("error:", Color.RED, bold=True)

  if severity == ir.DiagnosticSeverity.WARNING:
    return colored("warning:", Color.MAGENTA, bold=True)

  if severity == ir.DiagnosticSeverity.REMARK:
    return colored("remark:", Color.YELLOW, bold=True)

  if severity == ir.DiagnosticSeverity.NOTE:
    return colored("note:", Color.BLUE, bold=True)

  raise ValueError("unknown severity")


class Printer:
  """
  Helper class to print indented diagnostic messages.
  """

  def __init__(self, indent: int, verbosity: Verbosity = Verbosity.NORMAL):
    self.indent = indent
    self.traceback_indent = 0
    self.verbosity = verbosity

  def print(self, *args):
    print(colored("  |", Color.CYAN) *
          (self.indent + max(0, self.traceback_indent - 1)),
          *args,
          file=sys.stderr)

  def print_source_line(self, file_loc: ir.Location):
    try:
      with open(file_loc.filename, 'r') as f:
        lines = f.readlines()

        start_line = file_loc.start_line
        end_line = file_loc.end_line
        start_col = file_loc.start_col
        end_col = file_loc.end_col

        # Validate line numbers
        if not (0 <= start_line - 1 < len(lines)):
          return
        if not (0 <= end_line - 1 < len(lines)):
          end_line = start_line

        # Print each line in the range with appropriate indicators
        for line_num in range(start_line, end_line + 1):
          if 0 <= line_num - 1 < len(lines):
            source_line = lines[line_num - 1].rstrip()
            self.print(colored(" |", Color.CYAN), source_line)

            # Calculate indicator position and length
            if start_line == end_line:
              num_spaces = start_col
              num_carets = max(1, end_col - start_col)
            elif line_num == start_line:
              num_spaces = start_col
              num_carets = max(1, len(source_line) - start_col)
            elif line_num == end_line:
              num_spaces = 0
              num_carets = max(1, end_col)
            else:
              num_spaces = 0
              num_carets = max(1, len(source_line))

            indicator = ' ' * num_spaces + '^' * num_carets
            self.print(colored(" |", Color.CYAN), colored(indicator, Color.RED))
    except Exception:
      pass

  def print_traceback(self, trace: List[ir.Location]):
    self.traceback_indent += 1

    if self.verbosity != Verbosity.VERBOSE:
      trace = [loc for loc in trace if _is_user_code(loc)]

    if len(trace) > 1:
      self.print(colored(" Traceback (most recent call last):", Color.CYAN))

    for loc in reversed(trace):
      self.print_loc(loc)

    self.traceback_indent -= 1

  def print_loc(self, loc: ir.Location):
    if loc.is_a_name():
      self.print_loc(loc.child_loc)
    elif loc.is_a_callsite():
      trace = []
      while loc.is_a_callsite():
        trace.append(loc.callee)
        loc = loc.caller
      trace.append(loc)
      self.print_traceback(trace)
    elif loc.is_a_file():
      self.print(colored("--> ", Color.CYAN),
                 f"{loc.filename}:{loc.start_line}:{loc.start_col}")
      self.print_source_line(loc)
    elif loc == ir.Location.unknown():
      self.print(colored("--> ", Color.CYAN), "unknown location")
    else:
      raise ValueError("unexpected location type")


def get_diagnostic_handler(verbosity: Verbosity):
  """
  Returns a custom diagnostic handler for nicer error formatting.

  This handler processes MLIR diagnostics and formats them with colored output,
  source location information, and nested notes. It filters out internal
  locations and notes depending on the verbosity level and displays Python
  tracebacks when available.

  :param verbosity: Verbosity level for diagnostic output.
                    - CONCISE: Omit notes
                    - NORMAL: Show notes, filter internal locations
                    - VERBOSE: Show notes and internal locations
  :return: A diagnostic handler function that processes MLIR diagnostics.
  """

  def diagnostic_handler(diag, indent=0):
    """
    Internal diagnostic handler function.
    
    :param diag: The MLIR diagnostic to handle.
    :param indent: Indentation level for nested diagnostics (default: 0).
    :return: True to indicate the diagnostic has been processed.
    """

    # Skip printing the current operation (can be removed once the MLIR
    # context bindings support the option to disable automatic addition of
    # this note).
    if diag.message.startswith("see current operation"):
      return True

    p = Printer(indent, verbosity)
    p.print()
    p.print(colored_severity(diag.severity),
            colored(diag.message, Color.NONE, bold=True))
    p.print_loc(diag.location)

    if verbosity != Verbosity.CONCISE:
      for note in diag.notes:
        diagnostic_handler(note, indent=1)

    return True

  return diagnostic_handler
