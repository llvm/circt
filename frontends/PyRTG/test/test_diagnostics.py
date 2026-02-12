# RUN: %PYTHON% -u %s 2>&1 | FileCheck %s --strict-whitespace

import sys
import os
import tempfile
from io import StringIO
from pyrtg.diagnostics import (PySourceLocation, PySourceLocationStack, colored,
                               colored_severity, Printer, Color)
from pyrtg.base import ir


def test_py_source_location():
  # CHECK-LABEL: test_py_source_location
  print("test_py_source_location")

  with ir.Context():
    loc = PySourceLocation("test.py", 10, 5)
    # CHECK: loc("test.py":10:5)
    print(loc.to_ir_location())

    loc = PySourceLocation("test.py", 10, 5, end_line=10, end_col=15)
    # CHECK: loc("test.py":10:5 to :15)
    print(loc.to_ir_location())

    loc = PySourceLocation("test.py", 10, 5, end_line=12, end_col=8)
    # CHECK: loc("test.py":10:5 to 12:8)
    print(loc.to_ir_location())


def test_py_source_location_errors():
  # CHECK-LABEL: test_py_source_location_errors
  print("test_py_source_location_errors")

  with ir.Context():
    loc = PySourceLocation("test.py", 10, 5, end_line=12, end_col=None)
    try:
      loc.to_ir_location()
      # CHECK-NOT: should_not_reach_end_line_without_end_col
      print("should_not_reach_end_line_without_end_col")
    except ValueError as e:
      # CHECK: end_line is provided but end_col is not
      print(e)

    loc = PySourceLocation("test.py", 10, 5, end_line=8, end_col=15)
    try:
      loc.to_ir_location()
      # CHECK-NOT: should_not_reach_end_line_less_than_line
      print("should_not_reach_end_line_less_than_line")
    except ValueError as e:
      # CHECK: end_line is provided but is less than line
      print(e)


def test_py_source_location_stack():
  # CHECK-LABEL: test_py_source_location_stack
  print("test_py_source_location_stack")

  with ir.Context():
    stack = PySourceLocationStack([])
    # CHECK: loc(unknown)
    print(stack.to_ir_location())

    loc = PySourceLocation("test.py", 10, 5)
    stack = PySourceLocationStack([loc])
    # CHECK: loc("test.py":10:5)
    print(stack.to_ir_location())

    locs = [
        PySourceLocation("caller.py", 20, 10),
        PySourceLocation("main.py", 5, 0)
    ]
    stack = PySourceLocationStack(locs)
    # CHECK: loc(callsite("caller.py":20:10 at "main.py":5:0))
    print(stack.to_ir_location())

    locs = [
        PySourceLocation("level3.py", 30, 15),
        PySourceLocation("level2.py", 20, 10),
        PySourceLocation("level1.py", 10, 5),
        PySourceLocation("main.py", 5, 0)
    ]
    stack = PySourceLocationStack(locs)
    # CHECK: loc(callsite("level3.py":30:15 at callsite("level2.py":20:10 at callsite("level1.py":10:5 at "main.py":5:0)))
    print(stack.to_ir_location())


def test_colored_without_tty():
  # CHECK-LABEL: test_colored_without_tty
  print("test_colored_without_tty")

  # Mock stderr.isatty to return False
  old_isatty = sys.stderr.isatty
  sys.stderr.isatty = lambda: False

  result = colored("test", Color.RED)
  # CHECK: btesta
  print(f"b{result}a")

  sys.stderr.isatty = old_isatty


def test_colored_severity():
  # CHECK-LABEL: test_colored_severity
  print("test_colored_severity")

  result = colored_severity(ir.DiagnosticSeverity.ERROR)
  # CHECK: berror:a
  print(f"b{result}a")

  result = colored_severity(ir.DiagnosticSeverity.WARNING)
  # CHECK: bwarning:a
  print(f"b{result}a")

  result = colored_severity(ir.DiagnosticSeverity.NOTE)
  # CHECK: bnote:a
  print(f"b{result}a")

  result = colored_severity(ir.DiagnosticSeverity.REMARK)
  # CHECK: bremark:a
  print(f"b{result}a")

  try:
    colored_severity(999)
    # CHECK-NOT: should_not_reach
    print("should_not_reach")
  except ValueError as e:
    # CHECK: unknown severity
    print(e)


def test_printer():
  # CHECK-LABEL: test_printer
  print("test_printer")

  printer = Printer(indent=0)
  # CHECK-NEXT: test message
  printer.print("test message")

  printer = Printer(indent=2)
  # CHECK-NEXT: |  | test
  printer.print("test")


def test_printer_source_line():
  # CHECK-LABEL: test_printer_source_line
  print("test_printer_source_line")

  with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write("line 1\n")
    f.write("line 2 with some code\n")
    f.write("line 3\n")
    temp_path = f.name

  try:
    with ir.Context():
      # Test with single line, starting at column 0
      printer = Printer(indent=0)
      loc = PySourceLocation(temp_path, 2, 0).to_ir_location()
      # CHECK: | line 2 with some code
      # CHECK-NEXT: | ^{{$}}
      printer.print_source_line(loc)

      # Test with single line, column range
      printer = Printer(indent=0)
      loc = PySourceLocation(temp_path, 2, 11, end_line=2,
                             end_col=15).to_ir_location()
      # CHECK: | line 2 with some code
      # CHECK-NEXT: |            ^^^^{{$}}
      printer.print_source_line(loc)

      # Test with multiline, 3 lines
      printer = Printer(indent=0)
      loc = PySourceLocation(temp_path, 1, 5, end_line=3,
                             end_col=4).to_ir_location()
      # CHECK: | line 1
      # CHECK-NEXT: |      ^{{$}}
      # CHECK-NEXT: | line 2 with some code
      # CHECK-NEXT: | ^^^^^^^^^^^^^^^^^^^^^{{$}}
      # CHECK-NEXT: | line 3
      # CHECK-NEXT: | ^^^^{{$}}
      printer.print_source_line(loc)
  finally:
    os.remove(temp_path)


def test_printer_print_loc():
  # CHECK-LABEL: test_printer_print_loc
  print("test_printer_print_loc")

  with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write("line 1\n")
    f.write("line 2 with some code\n")
    f.write("line 3\n")
    temp_path = f.name

  try:
    with ir.Context():
      # Test with file location
      printer = Printer(indent=0)
      loc = PySourceLocation(temp_path, 2, 5).to_ir_location()
      # CHECK: --> {{.*}}:2:5
      # CHECK-NEXT: line 2 with some code
      printer.print_loc(loc)

      # Test with unknown location
      printer = Printer(indent=0)
      # CHECK: -->  unknown location
      printer.print_loc(ir.Location.unknown())

      # Test with name location
      printer = Printer(indent=0)
      file_loc = PySourceLocation(temp_path, 1, 0).to_ir_location()
      name_loc = ir.Location.name("test_name", file_loc)
      # CHECK: --> {{.*}}:1:0
      # CHECK-NEXT: line 1
      printer.print_loc(name_loc)

      # Test with single location (no traceback header)
      printer = Printer(indent=0)
      loc1 = PySourceLocation(temp_path, 1, 0).to_ir_location()
      # CHECK-NOT: Traceback
      # CHECK: --> {{.*}}:1:0
      # CHECK-NEXT: line 1
      printer.print_traceback([loc1])

      # Test with multiple locations (shows traceback header)
      printer = Printer(indent=0)
      loc1 = PySourceLocation(temp_path, 1, 0).to_ir_location()
      loc2 = PySourceLocation(temp_path, 2, 2).to_ir_location()
      loc3 = PySourceLocation(temp_path, 3, 0).to_ir_location()
      # CHECK: Traceback (most recent call last):
      # CHECK: --> {{.*}}:3:0
      # CHECK-NEXT: line 3
      # CHECK: --> {{.*}}:2:2
      # CHECK-NEXT: line 2 with some code
      # CHECK: --> {{.*}}:1:0
      # CHECK-NEXT: line 1
      printer.print_traceback([loc1, loc2, loc3])
  finally:
    os.remove(temp_path)


if __name__ == "__main__":
  test_py_source_location()
  test_py_source_location_errors()
  test_py_source_location_stack()
  test_colored_without_tty()
  test_colored_severity()
  test_printer()
  test_printer_source_line()
  test_printer_print_loc()
