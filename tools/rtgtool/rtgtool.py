#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
import argparse
import importlib.util
import sys
from pathlib import Path
from enum import Enum
import io
import os

def codegen(rtg, path):
  file = import_module_from_path(Path(path))

  module = rtg.ir.Module.create()
  with rtg.ir.InsertionPoint(module.body):
    for _, obj in inspect.getmembers(file):
      if hasattr(obj, '_is_scope') and obj._is_scope:
        obj(module)
      if hasattr(obj, '_is_target') and obj._is_target:
        obj(module.context)
      if hasattr(obj, '_is_test') and obj._is_test:
        obj(module.context)
  return module

def import_module_from_path(module_path):
  module_name = module_path.stem  # Removes the '.py' part

  # Load the module dynamically
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  
  return module

class InputFormat(Enum):
  MLIR = 'mlir'
  PYTHON = 'py'

class OutputFormat(Enum):
  MLIR = 'mlir'
  RENDERED_MLIR = 'rendered'
  ASM = 'asm'
  ELF = 'elf'


def parse_args():
  # Set up argument parser
  parser = argparse.ArgumentParser(description="Driver.")
  parser.add_argument("file", type=str, help="Path to the Python file to import")
  parser.add_argument("--libname", type=str, help="Python bindings module name")
  parser.add_argument("--seed", type=str, help="Seed for all RNGs during randomization")
  parser.add_argument("-o", type=str, help="Output file")
  parser.add_argument("--verify-each", type=bool, help="Run the verifier after each transformation pass", default=False)
  parser.add_argument("--verbose-pass-executions", type=bool, help="Log executions of toplevel module passes", default=False)
  parser.add_argument("--unsupported-instructions-file", type=str, help="File containing a comma-separated list of unsupported instruction names", default="")
  parser.add_argument("--unsupported-instructions", type=str, help="Comma-separated list of unsupported instruction names", default="")
  parser.add_argument("--input-format", type=InputFormat, choices=list(InputFormat), help="Input Format")
  parser.add_argument("--output-format", type=OutputFormat, choices=list(OutputFormat), help="Output Format", default=OutputFormat.ASM)
  
  # Parse command-line arguments
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()

  rtg = importlib.import_module(args.libname)

  with rtg.ir.Context() as ctx, rtg.ir.Location.unknown():
    rtg.register_dialects(ctx) 

    # If no input format is given, try to infer it from the file extension
    if args.input_format == None:
      filename, file_extension = os.path.splitext(args.file)
      if file_extension == ".mlir":
        args.input_format = InputFormat.MLIR
      elif file_extension == ".py":
        args.input_format = InputFormat.PYTHON

    if args.input_format == InputFormat.MLIR:
      with open(args.file, 'r') as f:
        module_op = rtg.ir.Module.parse(f.read())
    elif args.input_format == InputFormat.PYTHON:
      module_op = codegen(rtg, args.file)
    else:
      assert False, "input format must be one of the above"

    buffer = io.StringIO()
    seed = int(args.seed) if args.seed != None else 0
    rtg.generate_random_tests(module_op, args.verify_each, args.verbose_pass_executions, args.seed != None, seed, args.unsupported_instructions.split(','), args.unsupported_instructions_file, args.output_format.value, buffer)

    if args.o == None:
      print(buffer.getvalue(), end='')
    else:
      with open(args.o, 'w') as f:
        print(buffer.getvalue(), file=f, end='')
